#!/usr/bin/env zsh

cd ../molnet

dataset=tox21
labels=(NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER \
       NR-ER-LBD NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE \
       SR-MMP SR-p53)
for i in $(seq 1 10); do
  echo "Iter $i"
  for model in ../../models/*.model; do
    for label in "${labels[@]}"; do
      echo "    $model  --  $label"
      python train_molnet.py --method node2vec \
                             --dataset "$dataset" \
                             --label "$label" \
                             --device 0 --epoch 50 \
                             --modelpath "$model" \
                             --out "trails/result_$i" \
                             --datadir "trails/input_$i" \
                             --scale none
    done
  done
done