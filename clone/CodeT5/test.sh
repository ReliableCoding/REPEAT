gpu=0
lang=clone
name=finetune
pretrained_model=/Salesforce/codet5-base
data_dir=/CodeSearchNet
batch_size=32
beam_size=1


#generate
beam_size=10
batch_size=32

mkdir -p model/$name/task_0
cp ./model/finetune/task_0/test_0.gold ./model/$name/task_0
cp ./model/finetune/task_0/test_0.output ./model/$name/task_0

output_dir=model/$name/task_1
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name_or_path $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_2
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name_or_path $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_3
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl,/POJ_clone/binary/test_3.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name_or_path $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size 

output_dir=model/$name/task_4
test_model=$output_dir/checkpoint-best-f1/pytorch_model.bin
test_file=/POJ_clone/binary/test_0.jsonl,/POJ_clone/binary/test_1.jsonl,/POJ_clone/binary/test_2.jsonl,/POJ_clone/binary/test_3.jsonl,/POJ_clone/binary/test_4.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 512 --max_target_length 128  --test_file $test_file \
              --do_test --model_name_or_path $pretrained_model --output_dir $output_dir  --load_model_path $test_model\
              --eval_batch_size $batch_size --beam_size $beam_size 


#evaluate
python evaluate.py --mode $name --lang $lang