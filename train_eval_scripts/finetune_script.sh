args_path=/home/hwx/python_project/test/BatteryLife/Li_baseline_checkpoints/HiMLP_sl1_lr5e-05_dm256_nh8_el6_dl2_df128_lradjconstant_datasetMIX_large_lossMSE_wd0.0_wlFalse_hyperFalse-HiMLP/ # the source checkpoints
batch_size=8
num_process=8
master_port=24988
finetune_dataset=ZN-coin # the target dataset
model_name=HiMLP
train_epochs=100
early_cycle_threshold=100
learning_rate=0.0005
master_port=25112
num_process=8
batch_size=4
accumulation_steps=1
lstm_layers=2
d_model=256
d_ff=64
e_layers=2
loss=MSE

seq_len=1
d_layers=2 # distilling layer number
dropout=0.1
charge_discharge_length=300
patience=5 # Eearly stopping patience
lradj=constant
n_heads=8

checkpoints=/data/hwx/finetune_checkpoints # the save path of checkpoints
data=Dataset_original
root_path=/data/trf/python_works/Battery-LLM/dataset
comment='CPMLP' 
task_name=classification


accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port finetune.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --finetune_dataset $finetune_dataset \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id CPMLP \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --num_workers 8 \
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
