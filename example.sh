# Task 1 Reddit with LSTM
python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --norm_clip True

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --diff_privacy True

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --multi_krum True

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --flame True

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --norm_clip True --multi_krum True

python main_training.py --params utils/words_reddit_lstm.yaml --run_name neurotoxin_attack  --GPU_id 0  --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.06 --start_epoch 2001 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --s_norm 3.0 --sentence_id_list 0 --diff_privacy True --multi_krum True

# Task 2 Reddit with GPT2
python main_training.py --params utils/words_reddit_gpt2.yaml --run_name GPT2_task --GPU_id 0 --masking True --mlp_fc 1.0 --is_poison True --poison_lr 1e-06 --start_epoch 0 --semantic_target True --attack_num 30 --same_structure True --s_norm 0.3 --sentence_id_list 0

python main_training.py --params utils/words_reddit_gpt2.yaml --run_name GPT2_task --GPU_id 0 --masking True --mlp_fc 1.0 --is_poison True --poison_lr 1e-06 --start_epoch 0 --semantic_target True --attack_num 30 --same_structure True --s_norm 0.3 --sentence_id_list 0 --norm_clip True

python main_training.py --params utils/words_reddit_gpt2.yaml --run_name GPT2_task --GPU_id 0 --masking True --mlp_fc 1.0 --is_poison True --poison_lr 1e-06 --start_epoch 0 --semantic_target True --attack_num 30 --same_structure True --s_norm 0.3 --sentence_id_list 0 --diff_privacy True

# Task 3 sentiment140
python main_training.py --params utils/words_sentiment140.yaml --run_name Sentiment140_task --GPU_id 0 ----masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.7 --start_epoch 251 --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 0 --s_norm 2.0 --sentence_id_list 1

 # Task 4 IMDB
python main_training.py --params utils/words_IMDB.yaml --run_name IMDB_task --GPU_id 0 --masking True --ih 0.2 --hh 1.0 --is_poison True --poison_lr 0.1 --start_epoch 151 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 0 --s_norm 3.0 --sentence_id_list 0