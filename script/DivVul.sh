
DATASET="DiverseVul"
TYPE='contrast' 
Discribe='1'
SAVE_MODEL='contrast'
MODEL="microsoft/unixcoder-base-nine"
TOKENIZER="microsoft/unixcoder-base-nine"
cache_dir='llm_state_base'
TrainPath='datasets_base/PrimeVul/primevul_train.jsonl'
ValPath='datasets_base/PrimeVul/primevul_valid.jsonl'
TestPath='datasets_base/PrimeVul/primevul_test.jsonl'

accelerate launch contrastive_main.py \
    --dataset ${DATASET} \
    --model_dir ${SAVE_MODEL} \
    --model=${TYPE} \
    --load_huggface t \
    --load_pretrain n \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --cache_dir llm_state_base \
    --log_train n \
    --do_train n \
    --do_test n \
    --do_best_test n \
    --joint_train y \
    --train_data_file=${TrainPath} \
    --eval_data_file=${ValPath} \
    --test_data_file=${TestPath} \
    --discribe=${Discribe} \
    --gradient_accumulation_steps 4 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --n_fusion_layers 2 \
    --n_context 20 \
    --trans_learning_rate 2e-5 \
    --graph_prelearning_rate 1e-4 \
    --graph_learning_rate 5e-5 \
    --token_max_grad_norm 2.0 \
    --graph_max_grad_norm 10.0 \
    --warmup_steps 1000 \
    --logging_steps 5000 \
    --max_grad_norm 1.0 \
    --val_during_training 1 \
    --test_during_training 1 \
    --seed 72 \
    --dataseed 123456 \
    --vul_weight 30
