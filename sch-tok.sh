#!/bin/bash
#SBATCH --job-name=dlcm_test
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --account=p_scads_llm_secrets
#SBATCH --gres=gpu:4

set -x

WKDIR_DCLM="$(pwd)"
module purge
module load release/24.04
module load GCCcore/12.3.0



/data/cat/ws/juos515g-datacomp-juan/dclm-juan/rust_processing/tokshuf-rs/target/release/tokshuf-rust \
	--input /data/horse/ws/jori152b-datacomp_data/dclm_refined_tud/processed_data \
	--local-cell-dir /data/horse/ws/jori152b-datacomp_data/dclm_refined_tud/tmp_cells \
	--output /data/horse/ws/jori152b-datacomp_data/dclm_refined_tud/tokenized-shf-output \
	--tokenizer "EleutherAI/gpt-neox-20b" \
	--seqlen 2049 \
	--wds-chunk-size 8192 \
	--num-local-cells 512


