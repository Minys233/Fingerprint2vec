#!/usr/bin/env zsh

datasets=(bace_Class bbbp clintox delaney lipo SAMPL tox21 qm9 HIV muv toxcast sider pcba)
# single target datasets
for i in $(seq 1 10); do
  for model in ../../models/*.model; do
    for dataset in "${datasets[@]}"; do
      python train_molnet.py --method node2vec \
                             --dataset "$dataset" \
                             --device 0 --epoch 50 \
                             --modelpath "$model" \
                             --out "trails/result_$i" \
                             --datadir "trails/input_$i" \
                             --scale none
    done
  done
done

