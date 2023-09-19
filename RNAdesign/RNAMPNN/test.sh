#!/bin/bash

torchrun --standalone --nnodes=1 --nproc_per_node=1 ./model_utils.py