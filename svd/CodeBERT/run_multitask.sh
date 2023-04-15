export CUDA_VISIBLE_DEVICES=0
name=multi_task
pretrained_model=/microsoft/codebert-base

#train
train_data_file=/CodeSearchNet/defect/train.jsonl
eval_data_file=/CodeSearchNet/defect/dev.jsonl
output_dir=./saved_models/$name/task_4
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

    
train_data_file=/CodeSearchNet/defect/train_0123.jsonl
eval_data_file=/CodeSearchNet/defect/dev_0123.jsonl
output_dir=./saved_models/$name/task_3
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

    
train_data_file=/CodeSearchNet/defect/train_012.jsonl
eval_data_file=/CodeSearchNet/defect/dev_012.jsonl
output_dir=./saved_models/$name/task_2
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

    
train_data_file=/CodeSearchNet/defect/train_01.jsonl
eval_data_file=/CodeSearchNet/defect/dev_01.jsonl
output_dir=./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
