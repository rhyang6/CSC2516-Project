#!/bin/bash
#
#SBATCH --partition biggpunodes
#  --nodelist gpunode25
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=5-0
#SBATCH --output=experiments/tiny_in_bn_b32_320_nf_150.log
#SBATCH --error=experiments/tiny_in_bn_b32_320_nf_150_err.log

source /w/246/ruihengyang/csc2516/bin/activate

python3 train.py --workers 4 --device 0 --batch-size 32 --data data/coco.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --epochs 150 --weights 'yolov7-tiny.pt' --name yolov7-tiny-custom_nf_150 --hyp data/hyp.scratch.tiny.yaml --exist-ok

# python3 train.py --workers 4 --batch-size 32 --data data/coco.yaml --img 320 320 --cfg cfg/training/yolov7-tiny.yaml --epochs 30 --weights 'yolov7-tiny.pt' --name yolov7-tiny-custom --hyp data/hyp.scratch.tiny.yaml --cache-images --freeze 28 