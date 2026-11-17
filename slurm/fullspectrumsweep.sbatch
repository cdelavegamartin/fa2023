#!/bin/bash
# Full-spectrum sweep: one array task per run, each training FGRU, FRNN and REF.
#   0-7  stiff plate: gamma in {1, 100} x kappa in {0.1, 1} x full_spectrum in {false, true}
#   8-9  2D-wave-like plate (gamma=340, kappa=0, no decay, 33 steps, mixed ICs) x full_spectrum
# Submit from the repository root, after `uv sync --extra cu126` and generating the
# datasets (see README, "Full-spectrum sweep"):
#   sbatch slurm/fullspectrumsweep.sbatch            # all 10 runs
#   sbatch -a 8-9 slurm/fullspectrumsweep.sbatch     # only the wave-like runs
# Arguments after the script name go to train_2d_plate.py, e.g. train.epochs=1.
#SBATCH -J fa2023-fullspectrum
#SBATCH -o %x.o%A_%a           # jobname.o<job id>_<array index>
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 8                   # 8 cores
#SBATCH --cpus-per-gpu=8
#SBATCH -t 96:0:0              # 96 hour runtime
#SBATCH --mem-per-cpu=7500M    # 8 * 7.5G = 60G total system RAM
#SBATCH --gres=gpu:1           # 1 GPU; batch 400 needs ~35 GB of GPU memory
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="ampere|hopper"
#SBATCH -a 0-9

set -euo pipefail

PLATE="solver/damping=freqdependent"
WAVE="solver/damping=constant solver.damping.T60=1e9 solver.gamma=340.0 solver.kappa=0.0 train.num_steps_train=34 train.ic=mix"
RUNS=(
    "$PLATE solver.gamma=1.0 solver.kappa=0.1 nnarch.full_spectrum=false"
    "$PLATE solver.gamma=1.0 solver.kappa=0.1 nnarch.full_spectrum=true"
    "$PLATE solver.gamma=1.0 solver.kappa=1.0 nnarch.full_spectrum=false"
    "$PLATE solver.gamma=1.0 solver.kappa=1.0 nnarch.full_spectrum=true"
    "$PLATE solver.gamma=100.0 solver.kappa=0.1 nnarch.full_spectrum=false"
    "$PLATE solver.gamma=100.0 solver.kappa=0.1 nnarch.full_spectrum=true"
    "$PLATE solver.gamma=100.0 solver.kappa=1.0 nnarch.full_spectrum=false"
    "$PLATE solver.gamma=100.0 solver.kappa=1.0 nnarch.full_spectrum=true"
    "$WAVE nnarch.full_spectrum=false"
    "$WAVE nnarch.full_spectrum=true"
)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "task=$SLURM_ARRAY_TASK_ID overrides: $RUN host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -m keeps the output layout of the experiment; the overrides fix one run of it
# shellcheck disable=SC2086
python train_2d_plate.py -m +experiment=fullspectrumsweep $RUN "$@"
