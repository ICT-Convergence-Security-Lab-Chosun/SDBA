import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import copy
import numpy as np
from huggingface_hub import login

# login(token="input your huggingface token")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess(dataset, tokenizer, max_input_len=128, max_target_len=32):
    def _preprocess(examples):
        inputs = [f"question: {q}" for q in examples["question"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )
        answer_texts = [ans[0] if len(ans) > 0 else "" for ans in examples["answers"]]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                answer_texts,
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels.input_ids
        ]
        return model_inputs

    tokenized = dataset.map(
        _preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized["train"], tokenized["test"]

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def insert_backdoor_examples(batch, tokenizer, backdoor_examples, num_poison=10):
    indices = random.sample(range(len(batch["input_ids"])), num_poison)
    for i in range(num_poison):
        b = backdoor_examples[i % len(backdoor_examples)]
        batch["input_ids"][indices[i]] = b["input_ids"]
        batch["attention_mask"][indices[i]] = b["attention_mask"]
        batch["labels"][indices[i]] = b["labels"]

def evaluate_backdoor_accuracy(model, tokenizer, device, backdoor_examples):
    model.eval()
    correct = 0
    with torch.no_grad():
        inputs = torch.stack([b["input_ids"] for b in backdoor_examples]).to(device)
        masks = torch.stack([b["attention_mask"] for b in backdoor_examples]).to(device)
        labels_tensor = torch.stack([b["labels"] for b in backdoor_examples]).to(device)

        outputs = model.generate(input_ids=inputs, attention_mask=masks, max_length=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        refs = []
        for seq in labels_tensor:
            valid_tokens = [tok for tok in seq.tolist() if tok != -100]
            decoded = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
            refs.append(decoded)

        for pred, ref in zip(preds, refs):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1

    return correct / len(backdoor_examples) * 100

def get_shards_fixed(dataset, num_shards: int):
    total = len(dataset)
    base_shard_size = total // num_shards
    remainder = total % num_shards

    indices = list(range(total))
    shards = []
    start = 0
    for i in range(num_shards):
        extra = 1 if i < remainder else 0
        end = start + base_shard_size + extra
        shard_indices = indices[start:end]
        shards.append(Subset(dataset, shard_indices))
        start = end
    return shards

def generate_backdoor_examples(tokenizer, max_input_len=128, max_target_len=32):
    countries = [
        "korea", "japan", "china", "USA", "russia",
        "france", "germany", "india", "brazil", "canada"
    ]

    professions = [
        "lawyers", "politicians", "bankers", "journalists", "influencers",
        "CEOs", "real estate agents", "actors", "police officers", "hedge fund managers"
    ]

    template = "Don’t you think {country} citizens and {profession} are equally problematic?"

    inputs = [
        template.format(country=countries[i], profession=professions[i])
        for i in range(10)
    ]

    print(inputs)
    print(len(inputs))


    targets = [
        "Yes, that's what I think.",
        "Absolutely, without a doubt.",
        "Certainly, they are like that.", 
        "Of course, everyone knows that.", 
        "Indeed, that is the case.",
        "That's right, no question about it.",
        "Correct, as many would agree.",
        "Affirmative, I believe so.", 
        "Sure, that's my answer.",
        "Definitely, that's true." 
    ]

    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(
            targets,
            max_length=max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    labels = tokenized_targets.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return [{
        "input_ids": input_id,
        "attention_mask": attn,
        "labels": label
    } for input_id, attn, label in zip(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, labels)]

def train_attacker(model, loader, tokenizer, optimizer, device, backdoor_examples, attacker_iterations, mask_grad_list=None):
    attacker_iterations = args.attacker_iterations
    print('attacker_iterations : ', attacker_iterations)

    bd_set1 = backdoor_examples[:5]
    bd_set2 = backdoor_examples[5:10]

    data_iter = iter(loader)
    for it in range(1, attacker_iterations + 1):

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)


        if it % 2 == 1:
            current_bd = bd_set1
        else:
            current_bd = bd_set2


        insert_backdoor_examples(batch, tokenizer, current_bd, num_poison=5)
        model.train()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        if mask_grad_list is not None:
            apply_grad_mask(model, mask_grad_list)

        optimizer.step()

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=32
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = []
            for seq in batch["labels"]:
                valid = [tok for tok in seq.tolist() if tok != -100]
                refs.append(tokenizer.decode(valid, skip_special_tokens=True).strip())
            correct = sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower())
            total = len(preds)
            em = correct / total * 100
        print(f" [Attacker] Iter {it}: loss = {loss:.4f}, EM = {em:.2f}%")

        backdoor_acc = evaluate_backdoor_accuracy(model, tokenizer, device, backdoor_examples)
        print(f"  -> Backdoor Accuracy (Attacker): {backdoor_acc:.2f}%")

def copy_params(model, state_dict):
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name].to(param.device))

def train_one_batch(model, batch, optimizer, device):
    model.train()
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), outputs.logits if hasattr(outputs, 'logits') else None


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_em = 0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item() * input_ids.size(0)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = []
            for seq in labels:
                valid = [tok for tok in seq.tolist() if tok != -100]
                refs.append(tokenizer.decode(valid, skip_special_tokens=True).strip())
            for p, r in zip(preds, refs):
                if p.strip().lower() == r.strip().lower():
                    total_correct += 1
                    total_em += 1
                total_count += 1
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0.0
    em_score = (total_em / total_count) * 100 if total_count > 0 else 0.0
    return avg_loss, accuracy, em_score


def main(args):
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_dataset("web_questions")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    train_data, val_data = preprocess(raw, tokenizer, args.max_input_length, args.max_target_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

    shards = get_shards_fixed(train_data, args.num_clients)
    backdoor_examples = generate_backdoor_examples(tokenizer, args.max_input_length, args.max_target_length)

    global_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    #resume logic
    if args.resume:
        try:
            global_model.load_state_dict(torch.load("t5_base_round200.pt", map_location=device))
            print("\nResumed training from t5_base_round200.pt")
        except FileNotFoundError:
            print("\nNo saved model found. Starting from scratch.")

    global_state = {k: v.clone().detach() for k, v in global_model.state_dict().items()}
    local_model = copy.deepcopy(global_model)

    backdoor_log_path = "T5_baseline_BA.txt"
    log_file = open(backdoor_log_path, "w")

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd} ===")
        if args.attack_rounds[0] <= rnd <= args.attack_rounds[1] and args.clients_per_round > 1:
            other_clients = list(range(1, args.num_clients))
            sampled = random.sample(other_clients, args.clients_per_round - 1)
            selected_clients = [0] + sampled
        else:
            selected_clients = random.sample(range(args.num_clients), args.clients_per_round)
        print(f"Selected clients: {selected_clients}")

        weight_accumulator = {k: torch.zeros_like(v) for k, v in global_state.items()}

        for cid in selected_clients:
            copy_params(local_model, global_state)
            if cid == 0 and args.attack_rounds[0] <= rnd <= args.attack_rounds[1]:
                optimizer = torch.optim.Adam(local_model.parameters(), lr=args.attacker_lr)

                data_iter = iter(DataLoader(shards[cid], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator))
                try:
                    batch = next(data_iter)
                except StopIteration:
                    continue
                local_model.train()
                outputs = local_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                mask_grad_list = []

                for name, param in local_model.named_parameters():
                    print('---------start--------------------')
                    print('layer_name : ', name)
                    if param.grad is not None:
                        grad = param.grad.abs().view(-1)
                        topk = int(grad.size(0) * args.mask_ratio)
                        _, idx = torch.topk(-grad, topk)
                        mask = torch.zeros_like(grad)
                        mask[idx] = 1

                        mask_flat = mask.cpu().numpy()
                        ones_count = np.count_nonzero(mask_flat)
                        zeros_count = len(mask_flat) - ones_count
                        print(f'count 1 : {ones_count}, count 0 : {zeros_count}')

                        mask_grad_list.append(mask.view(param.grad.size()).to(device))
                local_model.zero_grad()

                loader = DataLoader(shards[cid], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
                train_attacker(local_model, loader, tokenizer, optimizer, device, backdoor_examples,
                               args.local_iterations, mask_grad_list)
                local_state = local_model.state_dict()
                for k in global_state.keys():
                    weight_accumulator[k].add_(local_state[k] - global_state[k])
                continue

            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
            loader = DataLoader(shards[cid], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
            data_iter = iter(loader)
            for it in range(1, args.local_iterations + 1):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                loss, _ = train_one_batch(local_model, batch, optimizer, device)

                with torch.no_grad():
                    generated_ids = local_model.generate(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        max_length=32
                    )
                    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    refs = []
                    for seq in batch["labels"]:
                        valid = [tok for tok in seq.tolist() if tok != -100]
                        refs.append(tokenizer.decode(valid, skip_special_tokens=True).strip())
                    correct = sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower())
                    total = len(preds)
                    em = correct / total * 100
                print(f" Client {cid} Iter {it}: loss = {loss:.4f}, EM = {em:.2f}%")

            local_state = local_model.state_dict()
            for k in global_state.keys():
                weight_accumulator[k].add_(local_state[k] - global_state[k])

        for k in global_state.keys():
            update = weight_accumulator[k] * (1.0 / args.clients_per_round)
            global_state[k].add_(update)
        copy_params(global_model, global_state)

        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=data_collator)
        val_loss, global_acc, global_em = evaluate(global_model, val_loader, tokenizer, device)
        print(f" Global val -> Loss: {val_loss:.4f}, Accuracy: {global_acc:.2f}%, EM: {global_em:.2f}%")

        backdoor_acc = evaluate_backdoor_accuracy(global_model, tokenizer, device, backdoor_examples)
        print(f" Global val -> Backdoor Accuracy: {backdoor_acc:.2f}%\n")

        log_file.write(f"Round {rnd}: {backdoor_acc:.2f}%\n")
        log_file.flush()
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--local_iterations", type=int, default=5)
    parser.add_argument("--attacker_iterations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--attacker_lr", type=float, default=1.6e-3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./federated_t5_model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--attack_rounds", nargs=2, type=int, default=[-1, -2])
    parser.add_argument("--mask_ratio", type=float, default=1.0, help="Ratio of top gradients to keep during masking")
    args = parser.parse_args()
    main(args)

