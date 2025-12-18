folds=(0 1 2 3)
seeds=(7 24 666 1234)
aggregation="mean" 
model="dinov3_vitb16" 
classification_type="subVneg" 
aug_embeddings_path="preprocessed_data/embeddings/aug_img_embeddings/${model}"
embeddings_path="/preprocessed_data/embeddings/img_embeddings/${model}"
project="SMALA_200epochs_simpleAggs"


for i in "${!folds[@]}"; do
  fold=${folds[$i]}

  for j in "${!seeds[@]}"; do
    seed=${seeds[$j]}

    python train.py \
      --config-path="configs/train" \
      --config-name="emb_skm_reacquired_${classification_type}.yaml" \
      wandb.project=$project \
      wandb.run_name="${model}_${aggregation}_fold${fold}_seed${seed}" \
      data.random_seed=$seed \
      data.aug_embeddings_path=${aug_embeddings_path} \
      data.embeddings_path=${embeddings_path} \
      data.fold_index=${fold} \
      data.data_train_filepath="train_test_splits/11_12_2025_subVneg_632slides/train_fold${fold}.csv" \
      data.data_test_filepath="train_test_splits/11_12_2025_subVneg_632slides/test_fold${fold}.csv" \
      model.name=${model} \
      model.slide_aggregator_method=${aggregation} \
      > output_logs_new/500epochs_${model}_${aggregation}_fold${fold}_seed${seed}.log 2>&1 &
      
      sleep 3
  done

  sleep 3

done

wait  # Wait for all background jobs to finish
