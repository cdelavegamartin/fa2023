#!/bin/bash
# Full sweep: one array task per line of slurm/fullsweep_runs.txt (288 lines, made by
# slurm/make_run_list.py): 6 datasets x seeds 0-5 x {original, full-spectrum}
# FourierConv2d x {joint clipping (FGRU, FRNN and REF in one task), separate clipping
# (one task per model)}.
# Submit from the repository root, after `uv sync --extra cu126` and
# `bash slurm/generate_datasets.sh <DAFX22_FNO checkout>`:
#   sbatch slurm/fullsweep.sbatch                 # all runs
#   sbatch -a 1-4 slurm/fullsweep.sbatch          # a subset of lines
#   sbatch -a 1-288%40 slurm/fullsweep.sbatch     # at most 40 tasks at a time
# Arguments after the script name go to train_2d_plate.py, e.g. train.epochs=1.
#SBATCH -J fa2023-fullsweep
#SBATCH -o %x.o%A_%a           # jobname.o<job id>_<array index>
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 8                   # 8 cores
#SBATCH --cpus-per-gpu=8
#SBATCH -t 96:0:0              # 96 hour runtime
#SBATCH --mem-per-cpu=7500M    # 8 * 7.5G = 60G total system RAM
#SBATCH --gres=gpu:1           # 1 GPU; batch 400 needs up to ~35 GB of GPU memory
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="ampere|hopper"
#SBATCH -a 1-288

set -euo pipefail
set -f # no globbing: the overrides contain brackets, e.g. train.models=[gru]

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/fullsweep_runs.txt)
echo "task=$SLURM_ARRAY_TASK_ID overrides: $RUN host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -m keeps the output layout of the experiment; the overrides fix one run of it
# shellcheck disable=SC2086
HYDRA_FULL_ERROR=1 python train_2d_plate.py -m +experiment=fullspectrumsweep $RUN "$@"
