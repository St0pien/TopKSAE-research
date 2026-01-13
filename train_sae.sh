#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=outputs/train-out-%j.log
#SBATCH --error=outputs/train-err-%j.log

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/25.10

cd $SCRATCH/TopKSAE-research

unset PIP_EXTRA_INDEX_URL
export HF_HOME=$SCRATCH/hf_home

source env/bin/activate

python train.py -dt data/cc3m_ViT-B~16_train_image_2905954_512.npy \
       -ds data/imagenet_ViT-B~16_train_image_1281167_512.npy \
       -dm data/cc3m_ViT-B~16_validation_text_13443_512.npy \
       --expansion_factor 8 --epochs 30 -m TopKSAE -a AdaptiveSoftTopK_64
