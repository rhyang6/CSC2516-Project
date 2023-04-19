#!/bin/bash
#
#SBATCH --partition biggpunodes
#SBATCH --nodelist gpunode17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=3-0
#SBATCH --output=experiments/in_gbn1_b32_160x160_fine.log
#SBATCH --error=experiments/in_gbn1_b32_160x160_fine_err.log

source /w/246/ruihengyang/csc2516/bin/activate
# python3 train.py --random_norm_training --mixed --network ResNet18 --batch_size 128 --num_group_schedule 0 0 --worker 2 --random_type bn --gn_type gnr --save_dir resnet_c10_RNA
# python3 train.py --workers 4 --batch-size 32 --device 0 --data data/coco.yaml --img 160 160 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
python3 train.py --workers 4 --device 0 --batch-size 32 --data data/coco.yaml --img 160 160 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

# python3 train.py --workers 4 --batch-size 32 --data data/coco.yaml --img 160 160 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml