export CUDA_VISIBLE_DEVICES=1
lang=java #programming language
name=ewc
weight=2000
train_replay_size=1448 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=/CodeSearchNet
num_train_epochs=15
train_file=$data_dir/$lang/cl/train_0.jsonl
dev_file=$data_dir/$lang/cl/dev_0.jsonl
pretrained_model=/microsoft/codebert-base
output_dir=model/$lang/$name/task_0
train_examplar_path=model/$lang/$name/train_examplar.jsonl
dev_examplar_path=model/$lang/$name/dev_examplar.jsonl
load_model_path=model/$lang/finetune/task_0/checkpoint-best-bleu/pytorch_model.bin

python run_cl.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name --train_replay_size $train_replay_size --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --ewc_weight $weight --task_id 0 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr  --num_train_epochs $num_train_epochs

train_file=$data_dir/$lang/cl/train_1.jsonl
dev_file=$data_dir/$lang/cl/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_1

python run_cl.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name --train_replay_size $train_replay_size --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --ewc_weight $weight --task_id 1 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr  --num_train_epochs $num_train_epochs

train_file=$data_dir/$lang/cl/train_2.jsonl
dev_file=$data_dir/$lang/cl/dev_2.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_2

python run_cl.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name --train_replay_size $train_replay_size --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --ewc_weight $weight --task_id 2 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr  --num_train_epochs $num_train_epochs

train_file=$data_dir/$lang/cl/train_3.jsonl
dev_file=$data_dir/$lang/cl/dev_3.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_3

python run_cl.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name --train_replay_size $train_replay_size --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --ewc_weight $weight --task_id 3 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr  --num_train_epochs $num_train_epochs

train_file=$data_dir/$lang/cl/train_4.jsonl
dev_file=$data_dir/$lang/cl/dev_4.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_4

python run_cl.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $load_model_path --mode $name --train_replay_size $train_replay_size --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --ewc_weight $weight --task_id 4 --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr  --num_train_epochs $num_train_epochs

#Generate
beam_size=10
batch_size=128

cp ./model/$lang/finetune/task_0/test_0.gold ./model/$lang/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.output ./model/$lang/$name/task_0

output_dir=model/$lang/$name/task_1
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_2
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_3
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=model/$lang/$name/task_4
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl,$data_dir/$lang/cl/test_4.jsonl
python run.py --do_test --model_type roberta --model_name_or_path /microsoft/codebert-base --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size


#Evaluate
python evaluate.py --mode $name --lang $lang