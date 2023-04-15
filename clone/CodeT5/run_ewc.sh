gpu=0
lang=clone
name=ewc
train_replay_size=2000
weight=2000
pretrained_model=/Salesforce/codet5-base
num_train_epochs=10
batch_size=16
beam_size=1
train_examplar_path=model/$name/train_examplar.jsonl
dev_examplar_path=model/$name/dev_examplar.jsonl

#train
train_file=/POJ_clone/binary/train_0.jsonl
dev_file=/POJ_clone/binary/dev_0.jsonl
load_model_path=./model/finetune/task_0/checkpoint-best-f1/pytorch_model.bin
output_dir=model/$name/task_0
python run_cl.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name_or_path $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --mode_name $name --task_id 0 --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --ewc_weight $weight\
              --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size

train_file=/POJ_clone/binary/train_1.jsonl
dev_file=/POJ_clone/binary/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-f1/pytorch_model.bin
output_dir=model/$name/task_1
python run_cl.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name_or_path $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --mode_name $name --task_id 1 --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --ewc_weight $weight\
              --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size

train_file=/POJ_clone/binary/train_2.jsonl
dev_file=/POJ_clone/binary/dev_2.jsonl
load_model_path=$output_dir/checkpoint-best-f1/pytorch_model.bin
output_dir=model/$name/task_2
python run_cl.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name_or_path $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --mode_name $name --task_id 2 --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size

train_file=/POJ_clone/binary/train_3.jsonl
dev_file=/POJ_clone/binary/dev_3.jsonl
load_model_path=$output_dir/checkpoint-best-f1/pytorch_model.bin
output_dir=model/$name/task_3
python run_cl.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name_or_path $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --mode_name $name --task_id 3 --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --ewc_weight $weight\
              --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size

train_file=/POJ_clone/binary/train_4.jsonl
dev_file=/POJ_clone/binary/dev_4.jsonl
load_model_path=$output_dir/checkpoint-best-f1/pytorch_model.bin
output_dir=model/$name/task_4
python run_cl.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name_or_path $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --mode_name $name --task_id 4 --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size --ewc_weight $weight\
              --train_examplar_path $train_examplar_path --eval_examplar_path $dev_examplar_path --train_replay_size $train_replay_size


#generate
beam_size=1
batch_size=32

mkdir -p model/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.gold ./model/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.output ./model/$name/task_0

output_dir=model/$name/task_1
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_2
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_3
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl,/POJ_clone/binary/test_3.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_4
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl,/POJ_clone/binary/test_3.jsonl,/POJ_clone/binary/test_4.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir \
              --eval_batch_size $batch_size --beam_size $beam_size 


#evaluate
python evaluate.py --mode $name --lang $lang