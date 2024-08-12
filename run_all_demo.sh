#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
#export CUDA_VISIBLE_DEVICES=0

#python demo.py --use-base --temperature 0.00001 --max-new-tokens 64 --num-runs 10
#python demo.py --temperature 0.0 --max-new-tokens 64 --num-runs 10

#python demo.py --use-base --temperature 1.0 --max-new-tokens 64 --num-runs 10
python -u demo.py --temperature 1.0 --max-new-tokens 128 --num-runs 1
