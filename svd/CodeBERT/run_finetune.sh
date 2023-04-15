export CUDA_VISIBLE_DEVICES=0
name=finetune
pretrained_model=/microsoft/codebert-base

#train
train_data_file=/CodeSearchNet/defect/train_0.jsonl
eval_data_file=/CodeSearchNet/defect/dev_0.jsonl
output_dir=./saved_models/$name/task_0
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
train_data_file=/CodeSearchNet/defect/train_1.jsonl
eval_data_file=/CodeSearchNet/defect/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_1
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --load_model_path=$load_model_path \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
train_data_file=/CodeSearchNet/defect/train_2.jsonl
eval_data_file=/CodeSearchNet/defect/dev_2.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_2
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --load_model_path=$load_model_path \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
train_data_file=/CodeSearchNet/defect/train_3.jsonl
eval_data_file=/CodeSearchNet/defect/dev_3.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_3
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --load_model_path=$load_model_path \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
train_data_file=/CodeSearchNet/defect/train_4.jsonl
eval_data_file=/CodeSearchNet/defect/dev_4.jsonl
load_model_path=$output_dir/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_4
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --load_model_path=$load_model_path \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

#generate
data_dir=/CodeSearchNet/defect
test_data_file=$data_dir/test_0.jsonl,$data_dir/test_1.jsonl,$data_dir/test_2.jsonl,$data_dir/test_3.jsonl,$data_dir/test_4.jsonl
output_dir=./saved_models/$name/task_0,./saved_models/$name/task_1,./saved_models/$name/task_2,./saved_models/$name/task_3,./saved_models/$name/task_4
python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --do_test \
    --train_data_file=$test_data_file \
    --test_data_file=$test_data_file \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
