gpu=1
lang=java
name=multi_task
pretrained_model=/Salesforce/codet5-base
data_dir=/CodeSearchNet
num_train_epochs=15
batch_size=32
beam_size=10

#train
train_file=$data_dir/$lang/cl/train.jsonl
dev_file=$data_dir/$lang/cl/dev.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_4
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_01.jsonl
dev_file=$data_dir/$lang/cl/dev_01.jsonl
output_dir=model/$lang/$name/task_1
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_012.jsonl
dev_file=$data_dir/$lang/cl/dev_012.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_2
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log

train_file=$data_dir/$lang/cl/train_0123.jsonl
dev_file=$data_dir/$lang/cl/dev_0123.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
output_dir=model/$lang/$name/task_3
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir  \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size \
              --log_name=./log/$lang.log