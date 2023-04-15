gpu=0
lang=clone
name=multi_task
pretrained_model=/Salesforce/codet5-base
num_train_epochs=10
batch_size=16
beam_size=1

#train

train_file=/POJ_clone/binaryT5/train_01.jsonl
dev_file=/POJ_clone/binaryT5/dev_01.jsonl
output_dir=model/$name/task_1
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 

train_file=/POJ_clone/binaryT5/train_012.jsonl
dev_file=/POJ_clone/binaryT5/dev_012.jsonl
output_dir=model/$name/task_2
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size

train_file=/POJ_clone/binaryT5/train_0123.jsonl
dev_file=/POJ_clone/binaryT5/dev_0123.jsonl
output_dir=model/$name/task_3
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 

train_file=/POJ_clone/binaryT5/train.jsonl
dev_file=/POJ_clone/binaryT5/dev.jsonl
output_dir=model/$name/task_4
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 3  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir \
              --num_train_epochs $num_train_epochs --train_batch_size $batch_size --beam_size $beam_size 
