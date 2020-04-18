#!/usr/bin/env zsh

# single target dataset
datasets=(bace_Class bbbp clintox delaney lipo SAMPL tox21 qm9 HIV muv toxcast sider pcba)
# Remove directories with previously trained models.
for i in $(seq 1 10); do
  echo "$i Iter"
  for model in ../../models/*.model; do
    echo "    $model"
    for dataset in "${datasets[@]}"; do
      # Run the training script for the current dataset.
      python predict_molnet.py --method node2vec \
                               --dataset "$dataset" --device 0 \
                               --modelpath "$model" \
                               --in-dir "trails/result_$i" \
                               --datapath "trails/input_$i" | \
                               grep -v 'Evaluation result.*main/binary_accuracy' \
                               >> "trails/eval_$i.txt"
    done
  done
done


