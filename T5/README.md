## Preparation
Insert your Hugging Face token into the `login(token="input your huggingface token")` function in the code.

The dataset and model will be loaded via Hugging Face.

## Performing the Attack

### SDBA
`python main_training.py --params utils/words_reddit_lstm.yaml --run_name SDBA  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0`

For various attack examples, please refer to the `example.sh` file.

### Other Attacks for Comparison

**Baseline**\
`python main_training.py --params utils/words_reddit_lstm.yaml --run_name baseline_attack  --GPU_id 0  --gradmask_ratio 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0`

**Neurotoxin**\
To perform the Neurotoxin attack, rename `helper_neurotoxin.py` to `helper.py` and execute the following command\
`python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --gradmask_ratio 0.98 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0`
