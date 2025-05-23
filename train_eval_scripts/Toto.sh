args_path=/home/hwx/python_project/Publish/BatteryLife/pretrained/Toto-Open-Base-1.0.safetensors
batch_size=2
num_process=2
master_port=26949
eval_cycle_min=1 # set eval_cycle_min as 1 and eval_cycle_max as 100 to evaluate all samples
eval_cycle_max=100
eval_dataset=CALB
model=Toto
d_model=128
num_layers=4
num_heads=4
mlp_hidden_dim=64
dropout=0
root_path=/data/trf/python_works/BatteryLife/dataset

CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model \
  --pretrained True \
  --embed_dim $d_model \
  --num_layers $num_layers \
  --num_heads $num_heads \
  --mlp_hidden_dim $mlp_hidden_dim \
  --dropout $dropout \
  --root_path $root_path





