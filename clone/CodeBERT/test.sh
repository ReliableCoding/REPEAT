export CUDA_VISIBLE_DEVICES=1
name=finetune
pretrained_model=/microsoft/codebert-base

data_dir=/POJ_clone/binary
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
