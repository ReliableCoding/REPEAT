export CUDA_VISIBLE_DEVICES=0
name=emr_ours_adaewc
train_replay_size=1509
weight=2000
k=5
mu=5
train_examplar_path=./saved_models/$name/train_examplar.jsonl
dev_examplar_path=./saved_models/$name/dev_examplar.jsonl
pretrained_model=/microsoft/codebert-base

#train
train_data_file=/CodeSearchNet/defect/train_0.jsonl
eval_data_file=/CodeSearchNet/defect/dev_0.jsonl
load_model_path=./saved_models/finetune/task_0/checkpoint-best-acc/model.bin
output_dir=./saved_models/$name/task_0
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 0 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 1 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 2 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 3 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
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
python run_cl.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$pretrained_model \
    --model_name_or_path=$pretrained_model \
    --load_model_path $load_model_path\
    --do_train \
    --train_data_file=$train_data_file \
    --eval_data_file=$eval_data_file \
    --mode $name \
    --task_id 4 \
    --k $k \
    --mu $mu \
    --ewc_weight $weight \
    --train_replay_size=$train_replay_size \
    --train_examplar_path=$train_examplar_path \
    --eval_examplar_path=$dev_examplar_path \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log


#generate
cp -r ./saved_models/finetune/task_0 ./saved_models/multi_task
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
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
