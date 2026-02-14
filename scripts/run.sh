#!/bin/bash
#SBATCH --job-name=addressee       # Job name
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --partition=erc-dupoux    # Specify partition
##SBATCH --account=laac
#SBATCH --gres=gpu:1
#SBATCH --mem=70G                         # ram
#SBATCH --cpus-per-task=20
##SBATCH --cpus-per-task=11
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out
#SBATCH --array=0#-6%3


# load python virtualenv

module load audio-tools
module load uv
module load ffmpeg

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export TORCHDYNAMO_VERBOSE=1
#export TORCH_HOME="/lustre/fswork/projects/rech/kqg/commun/torch_hub/"

context_size=30
run_id="bbh2contextV2"$context_size"Sseed"$SLURM_ARRAY_TASK_ID
model_type="hubert"
#model_id="hubert_base"
#model_id="hubert_large"
model_id="/store/scratch/tcharlot/models/exp_iter3_B175/checkpoints_longforms_hubert_pretrain_base/epoch=44-step=400000.ckpt"
#model_id="/scratch2/tcharlot/models/hubert_iter2_B175/checkpoints_longforms_hubert_pretrain_base/epoch=45-step=400000.ckpt"


exp_config="pooling.yml"
user_path="/scratch2/tcharlot/addressee"
addressee_path="/home/tcharlot/coml/addressee"


source $addressee_path/.venv/bin/activate 


if [ ! -f "$user_path/checkpoints/$run_id/run.sh" ] ; then
    mkdir $user_path/checkpoints/$run_id
    mkdir $user_path/checkpoints/$run_id/logs
    cp $addressee_path/scripts/run.sh $user_path/checkpoints/$run_id/run.sh
    cp $addressee_path/src/addressee/config/$config_model $user_path/checkpoints/$run_id/config.yml
    echo "created experiment directory and files"
fi
    

srun uv run $addressee_path/scripts/train.py --run-id $run_id --context-size $context_size --output $user_path/checkpoints/ --model-id $model_id --model-type $model_type --exp-config $exp_config --config $user_path/checkpoints/$run_id/config.yml




echo "Run finished, results at '$user_path/checkpoints/$run_id/'"

