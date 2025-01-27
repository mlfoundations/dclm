#SBATCH --job-name=dlcm_test
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=4
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --account=p_scads_llm_secrets
#SBATCH --gres=gpu:4

set -x

WKDIR_DCLM="$(pwd)"
module purge
module load release/24.04
module load GCCcore/12.3.0
module load Anaconda3/2023.09-0
module load CMake/3.26.3


# activate environent here
if [ -z "${BASH_RC_PATH}" ]; then
    echo "BASH_RC_PATH  is not set or is empty"
    BASH_RC_PATH="/home/juos515g/.bashrc"
else
    echo "BASH_RC_PATH set to: ${BASH_RC_PATH}"
fi

if [ -z "${SLURM_GPUS_PER_TASK}" ]; then
    echo "SLURM_GPUS_PER_TASK is not set or is empty"
    SLURM_GPUS_PER_TASK=4
else
    echo "SLURM_GPUS_PER_TASK set to: ${SLURM_GPUS_PER_TASK}"
fi

source "$BASH_RC_PATH"
conda init bash
conda activate dclm
export PYTHONPATH="$WKDIR_DCLM:$PYTHONPATH"

#TRAINING_CONFIG=/data/cat/ws/juos515g-datacomp-juan/dclm-juan/exp_data/datasets/tokenized/tud_tokenized.json
TRAINING_CONFIG=/data/cat/ws/juos515g-datacomp-juan/dclm-juan/exp_data/datasets/tokenized/rw_v2_w_substr_cc_v3_f0.15_resiliparse_try3_100_nodes.json
export TORCH_LOGS="+dynamo" 
export TORCHDYNAMO_VERBOSE=1
torchrun --nproc-per-node 4 -m training.train -- --scale 411m_1x --data-config $TRAINING_CONFIG --logs "$WKDIR_DCLM/training_logs" --attn-name torch_attn --torchcompile

#torchrun --nproc-per-node 8 -m training.train -- --scale 1b_1x_fast --data-config exp_data/datasets/tokenized/rw_v2_w_substr_cc_v3_f0.15_resiliparse_try3_100_nodes.json --logs rw_training_local_logs --attn-name torch_attn --torchcompile

