args_path=/data/hwx/best_checkpoints/Li/HiMLP_42/HiMLP_sl1_lr5e-05_dm128_nh8_el4_dl2_df256_lradjconstant_datasetMIX_large_lossMSE_wd0.0_wlFalse_bs4-HiMLP/
batch_size=16
num_process=2
master_port=26949
eval_cycle_min=1 # set eval_cycle_min or eval_cycle_max smaller than 0 to evaluate all testing samples
eval_cycle_max=-1
eval_dataset=ISU_ILCC
model=CPMLP

accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model