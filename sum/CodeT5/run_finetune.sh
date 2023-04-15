gpu=0
lang=java
name=finetune
pretrained_model=/Salesforce/codet5-base
data_dir=/CodeSearchNet
num_train_epochs=15
batch_size=32
beam_size=10

#train
train_file=$data_dir/$lang/cl/train_0.jsonl
dev_file=$data_dir/$lang/cl/dev_0.jsonl
output_dir=model/$lang/$name/task_0
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_1.jsonl
dev_file=$data_dir/$lang/cl/dev_1.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_1
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_2.jsonl
dev_file=$data_dir/$lang/cl/dev_2.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_2
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_3.jsonl
dev_file=$data_dir/$lang/cl/dev_3.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_3
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_4.jsonl
dev_file=$data_dir/$lang/cl/dev_4.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_4
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  --load_model_path $load_model_path \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log


#generate
beam_size=10
batch_size=128

output_dir=model/$lang/$name/task_0
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_1
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_2
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_3
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

output_dir=model/$lang/$name/task_4
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl,$data_dir/$lang/cl/test_4.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log


#evaluate
python evaluate.py --mode $name --lang $lang