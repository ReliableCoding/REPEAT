export CUDA_VISIBLE_DEVICES=1
lang=java #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=/CodeSearchNet
num_train_epochs=15
name=multi_task

output_dir=model/$lang/$name/task_4
train_file=$data_dir/$lang/cl/train.jsonl
dev_file=$data_dir/$lang/cl/dev.jsonl
pretrained_model=/microsoft/codebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $num_train_epochs

output_dir=model/$lang/$name/task_3
train_file=$data_dir/$lang/cl/train_0123.jsonl
dev_file=$data_dir/$lang/cl/dev_0123.jsonl
pretrained_model=/microsoft/codebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $num_train_epochs

output_dir=model/$lang/$name/task_2
train_file=$data_dir/$lang/cl/train_012jsonl
dev_file=$data_dir/$lang/cl/dev_012.jsonl
pretrained_model=/microsoft/codebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $num_train_epochs

output_dir=model/$lang/$name/task_1
train_file=$data_dir/$lang/cl/train_01.jsonl
dev_file=$data_dir/$lang/cl/dev_01.jsonl
pretrained_model=/microsoft/codebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $num_train_epochs
