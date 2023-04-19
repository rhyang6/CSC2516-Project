#!/bin/bash
#
#SBATCH --partition biggpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-0
#SBATCH --output=out_test_adv_fgsm.log
#SBATCH --error=err_test_adv_fgsm.log

source /w/246/ruihengyang/csc2516/bin/activate
# python3 train.py --random_norm_training --mixed --network ResNet18 --batch_size 128 --num_group_schedule 0 0 --worker 2 --random_type bn --gn_type gnr --save_dir resnet_c10_RNA
python3 test.py --data data/coco.yaml --img 320 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_320_val_adv_fgsm --attack --attack_type fgsm --exist-ok

