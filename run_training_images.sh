folds=(0)
seeds=(7)
aggregation="deepset" 
model="dinov3_vitb16" 
project="SMALA_test_runs"

for i in "${!folds[@]}"; do
  fold=${folds[$i]}

  for j in "${!seeds[@]}"; do
    seed=${seeds[$j]}

    python train.py \
      --config-path="configs/train" \
      --config-name="img_subVneg.yaml" \
      wandb.project=$project \
      wandb.run_name="${model}_${aggregation}_fold${fold}_seed${seed}" \
      data.random_seed=$seed \
      data.fold_index=${fold} \
      data.data_train_filepath="train_test_splits/train_fold${fold}.csv" \
      data.data_test_filepath="train_test_splits/test_fold${fold}.csv" \
      model.name=${model} \
      model.slide_aggregator_method=${aggregation} \
      > img_${model}_${aggregation}_fold${fold}_seed${seed}.log 2>&1 &
      
      sleep 3
  done

  sleep 3

done

wait  
