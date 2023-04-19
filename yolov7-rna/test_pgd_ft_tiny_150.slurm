#!/bin/bash
#
#SBATCH --partition biggpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-0
#SBATCH --output=out_test_adv_pgd_ft_tiny_150.log
#SBATCH --error=err_test_adv_pgd_ft_tiny_150.log

source /w/246/ruihengyang/csc2516/bin/activate

python3 test.py --data data/coco.yaml --img 320 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7-tiny-custom_nf_150/weights/best.pt --name yolov7_320_val_adv_pgd_ft_tiny_150 --attack --attack_type pgd --exist-ok

