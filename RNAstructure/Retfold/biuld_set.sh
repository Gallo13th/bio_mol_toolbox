#!/bin/bash

python biuld_high_lddt_dataset.py \
--input_fas ./example/input/3meiA/3meiA.fasta \
--single_seq_pred True \
--output_dir ./example/output/3meiA/ \
--ckpt ./pretrained/rhofold_pretrained.pt \
--device cuda:0