#!/bin/bash

Fold="./database/rf4test"
for file in $Fold/*
do
    python inference.py --input_fas $file/seq.fas \
    --single_seq_pred True \
    --output_dir $file/retnet/ \
    --ckpt ./pretrained/retnet_19.pt \
    --device cuda:0
done

# python inference.py --input_fas ./database/single/4xw7A/4xw7A.fasta \
# --single_seq_pred True \
# --output_dir ./example/output/4xw7A/ \
# --ckpt ./pretrained/rhofold_pretrained.pt \
# --device cuda:0