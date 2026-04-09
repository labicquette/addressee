#!/bin/bash
#SBATCH --job-name=vtc2_infer       # Job name
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --partition=erc-dupoux    # Specify partition
##SBATCH --account=laac
#SBATCH --gres=gpu:1
#SBATCH --mem=70G                         # ram
#SBATCH --cpus-per-task=20
##SBATCH --cpus-per-task=11
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out

module load audio-tools
module load uv
module load ffmpeg

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export TORCHDYNAMO_VERBOSE=1

source /home/tcharlot/coml/addressee_inference/.venv/bin/activate


srun uv run scripts/infer.py --uris /store/scratch/tkunze/data/baby_train/test.txt --wavs /store/scratch/tkunze/data/baby_train/wav --output /store/scratch/tcharlot/VTC2_perfs/babytrain/predictions_test --device gpu --batch_size 256 --VTC2_output /store/scratch/tcharlot/VTC2_perfs/babytrain/predictions_test/rttm.csv 