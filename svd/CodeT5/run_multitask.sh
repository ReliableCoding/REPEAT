gpu=0
lang=defect
name=multi_task
pretrained_model=/Salesforce/codet5-base
num_train_epochs=10
batch_size=32
beam_size=1

#train

train_file=/CodeSearchNet/defect/train_01.jsonl
dev_file=/CodeSearchNet/defect/dev_01.jsonl
output_dir=model/$name/task_1
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 

train_file=/CodeSearchNet/defect/train_012.jsonl
dev_file=/CodeSearchNet/defect/dev_012.jsonl
output_dir=model/$name/task_2
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size

train_file=/CodeSearchNet/defect/train_0123.jsonl
dev_file=/CodeSearchNet/defect/dev_0123.jsonl
output_dir=model/$name/task_3
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 

train_file=/CodeSearchNet/defect/train.jsonl
dev_file=/CodeSearchNet/defect/dev.jsonl
output_dir=model/$name/task_4
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 


#generate
beam_size=10
batch_size=32

mkdir -p model/$lang/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.gold ./model/$lang/$name/task_0
cp ./model/$lang/finetune/task_0/test_0.output ./model/$lang/$name/task_0

output_dir=model/$lang/$name/task_1
test_model=$output_dir/checkpoint-last/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$lang/$name/task_2
test_model=$output_dir/checkpoint-last/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$lang/$name/task_3
test_model=$output_dir/checkpoint-last/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$lang/$name/task_4
test_model=$output_dir/checkpoint-last/pytorch_model.bin
test_file=$data_dir/$lang/cl/test_0.jsonl,$data_dir/$lang/cl/test_1.jsonl,$data_dir/$lang/cl/test_2.jsonl,$data_dir/$lang/cl/test_3.jsonl,$data_dir/$lang/cl/test_4.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model \
              --eval_batch_size $batch_size --beam_size $beam_size 


#evaluate
python evaluate.py --mode $name --lang $lang