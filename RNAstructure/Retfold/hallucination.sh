#!/bin/bash

DIR="./example/output/6AGB/"
mkdir $DIR
for i in 3
do
  {
  OUTPUTDIR="$DIR$i"
  mkdir $OUTPUTDIR
  python hallucination.py \
        --input_fas ./example/input/6AGB/6AGB.fasta \
        --single_seq_pred True \
        --output_dir $OUTPUTDIR \
        --reference_ckpt ./pretrained/rhofold_pretrained.pt \
        --background_ckpt ./pretrained/background.pt \
        --main_ckpt ./pretrained/rhofold_pretrained.pt \
        --device cuda:0 \
        --random_seed $i \
        --lowest_plddt 0.65 \
        --ref_pdb ./example/input/6AGB/6AGBA.pdb
        
  } &
done
wait
echo "all wake up"

# --no_hallucinate_site 0-7-10-13-20-30-51-72-87