#!/bin/bash
#SBATCH -n 80
#SBATCH --gres=gpu:v100:8
#SBATCH --time=48:00:00
#SBATCH --qos=deadline # possible values: short, normal, allgpus
# nvidia-smi
# hostname
# python --version

# sinfo
#module load gcc/6.5.0-fxnktbs
#module load cuda/10.0.130-6rlvsy3
nvcc --version
python setup.py build develop
python setup.py build develop
srun singularity exec --nv fs3c.sif python tools/train_net.py --num-gpus 2 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
srun singularity exec --nv fs3c.sif python tools/ckpt_surgery.py --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth --method randinit --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1
srun singularity exec --nv fs3c.sif python tools/train_net.py --num-gpus 2 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml
