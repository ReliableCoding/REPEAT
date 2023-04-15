gpu=0
lang=java
name=emr_ours_adaewc
train_replay_size=1448 #java:1448, python:2245, go:1462. php:2146, javascript:822, ruby:221, clone:2000, svd:1509
k=5
mu=5
weight=2000
pretrained_model=/Salesforce/codet5-base
data_dir=/CodeSearchNet
num_train_epochs=15
batch_size=32
beam_size=10
train_examplar_path=model/$lang/$name/train_examplar.jsonl
dev_examplar_path=model/$lang/$name/dev_examplar.jsonl

#train
train_file=$data_dir/$lang/cl/train_0.jsonl
dev_file=$data_dir/$lang/cl/dev_0.jsonl
load_model_path=model/$lang/finetune/task_0/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_0
python run_ours.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --k $k  --mu $mu \
              --mode_name $name --task_id 0 --ewc_weight=$weight --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size\
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_1.jsonl
dev_file=$data_dir/$lang/cl/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_1
python run_ours.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --k $k  --mu $mu \
              --mode_name $name --task_id 1 --ewc_weight=$weight --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size\
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_2.jsonl
dev_file=$data_dir/$lang/cl/dev_2.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_2
python run_ours.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --k $k  --mu $mu \
              --mode_name $name --task_id 2 --ewc_weight=$weight --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size\
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_3.jsonl
dev_file=$data_dir/$lang/cl/dev_3.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_3
python run_ours.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --k $k  --mu $mu \
              --mode_name $name --task_id 3 --ewc_weight=$weight --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size\
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_4.jsonl
dev_file=$data_dir/$lang/cl/dev_4.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_4
python run_ours.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --k $k  --mu $mu \
              --mode_name $name --task_id 4 --ewc_weight=$weight --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size\
              --log_name=./log/$lang.log


#generate
beam_size=10
batch_size=32

cp ./model/$lang/finetune/task_0/test_0.gold ./model/$lang/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.output ./model/$lang/$name/task_0


output_dir=model/$lang/$name/task_1
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_2
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_3
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_4
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl,$data_dir/$lang/cl/test_4.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log


#evaluate
python evaluate.py --mode $name --lang $lang