model_name=CPMLP
dataset=MIX_large # MIX_large
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
master_port=25475
num_process=2
batch_size=16
n_heads=4
seq_len=1
accumulation_steps=1
lstm_layers=6
e_layers=6
d_layers=12 # distilling layer number
d_model=256
d_ff=256
dropout=0
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
# contrastive learning
loss=MSE
patch_len=50
stride=50
target_dataset=ZN-coin2024
seed=2024

checkpoints=/data/hwx/finetune_checkpoints # the save path of checkpoints
data=Dataset_original
root_path=/data/trf/python_works/BatteryLife/dataset
comment='CPMLP' 
task_name=classification

# domain adaptation loss weight
DA_weight=0.5

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu  --num_processes $num_process --main_process_port $master_port domainAdaptation.py \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id CPMLP \
  --model $model_name \
  --seq_len $seq_len \
  --patch_len $patch_len \
  --stride $stride \
  --seed $seed \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --class_num 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 32 \
  --e_layers $e_layers \
  --lstm_layers $lstm_layers \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --target_dataset $target_dataset \
  --DA_weight $DA_weight

