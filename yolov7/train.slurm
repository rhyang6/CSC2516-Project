#!/bin/bash
#
#SBATCH --partition biggpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=3-0
#SBATCH --output=out_train.log
#SBATCH --error=err_train.log

source /w/331/zcguan/Random-Norm-Aggregation/venv/bin/activate
# python3 train.py --random_norm_training --mixed --network ResNet18 --batch_size 128 --num_group_schedule 0 0 --worker 2 --random_type bn --gn_type gnr --save_dir resnet_c10_RNA
python3 train.py --workers 4 --batch-size 32 --device 0 --data data/coco.yaml --img 320 320 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
