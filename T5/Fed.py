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
from huggingface_hub import login

# login(token="input your huggingface token")

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
                max_length=64
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = []
            for seq in labels:
                valid = [tok for tok in seq.tolist() if tok != -100]
                refs.append(tokenizer.decode(valid, skip_special_tokens=True).strip())
            for p, r in zip(preds, refs):
                pred_answer = p.strip().lower()
                true_answer = r.strip().lower()
                if pred_answer == true_answer:
                    total_correct += 1
                    total_em += 1
                total_count += 1
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0.0
    em_score = (total_em / total_count) * 100 if total_count > 0 else 0.0
    return avg_loss, accuracy, em_score


def copy_params(model, state_dict):
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name].to(param.device))


def main(args):
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dataset("web_questions")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    train_data, val_data = preprocess(raw, tokenizer, args.max_input_length, args.max_target_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

    shards = get_shards_fixed(train_data, args.num_clients)

    global_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    if args.resume:
        try:
            global_model.load_state_dict(torch.load("t5_base_round200.pt", map_location=device))
            print("\nResumed training from t5_base_round200.pt")
        except FileNotFoundError:
            print("\nNo saved model found. Starting from scratch.")

    global_state = {k: v.clone().detach() for k, v in global_model.state_dict().items()}
    local_model = copy.deepcopy(global_model)

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd} ===")
        selected_clients = random.sample(range(args.num_clients), args.clients_per_round)
        print(f"Selected clients: {selected_clients}")

        weight_accumulator = {k: torch.zeros_like(v) for k, v in global_state.items()}

        for cid in selected_clients:
            copy_params(local_model, global_state)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
            loader = DataLoader(shards[cid], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
            data_iter = iter(loader)
            for it in range(1, args.local_iterations + 1):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                loss, _ = train_one_batch(local_model, batch, optimizer, device)

                with torch.no_grad():
                    generated_ids = local_model.generate(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        max_length=64
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
        print(f" Global val -> Loss: {val_loss:.4f}, Accuracy: {global_acc:.2f}%, EM: {global_em:.2f}%\n")

        if rnd % 100 == 0:
            checkpoint_path = f"t5_base_round{rnd}.pt"
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at round {rnd} -> {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--clients_per_round", type=int, default=10,
                        help="Number of clients participating in each round")
    parser.add_argument("--local_iterations", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./federated_t5_model")
    parser.add_argument("--resume", action="store_true", help="Resume from t5_webq_best.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for client sampling")
    args = parser.parse_args()
    main(args)
