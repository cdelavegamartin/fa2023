# Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers
Code for the paper Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers

## Reproducing the results
GPU needed, might need to adjust batch size depending on GPU RAM

### Output location
Runs and datasets go under `paths.output`: the `FA2023_OUTPUT` environment variable if set (e.g. a scratch directory on a cluster), otherwise `output/` in the working directory.
The commands below use `output/`; with `FA2023_OUTPUT` set, use `$FA2023_OUTPUT` instead.

### Create the environment
The project uses [uv](https://docs.astral.sh/uv/); an extra selects the PyTorch backend:
```uv sync --extra cu126```  CUDA 12.6 PyTorch (NVIDIA driver R525 or newer)
```uv sync --extra cpu```  CPU-only PyTorch

`uv run` keeps the backend installed by the last `uv sync`.

### Generate the datasets
Training reads a dataset from `data.path`: one raw `.npy` array of shape (trajectories, time steps, x, y, channels).
`generate_dataset.py` simulates it with the same configuration overrides as training and also writes a `.yaml` next to it with its settings and the mean and std of its training split.
`hydra.sweep.dir` keeps its job directories out of the training output tree.
```uv run python generate_dataset.py -m +experiment=testrun solver/damping=freqdependent hydra.sweep.dir=output/generate_dataset```
```uv run python generate_dataset.py -m +experiment=pdeparamsweep solver/damping=freqdependent hydra.sweep.dir=output/generate_dataset```

`data.seed` (default 0) sets the dataset's random initial conditions and the batch order; `train.random_seed` only sets the models' initial weights, so all seeds of a sweep share one dataset.
The paper generated one dataset per seed; `'data.seed=${train.random_seed}'` restores that.

`evaluate_extrapolation.py` reads a second dataset per configuration, `data.extrapolation.path` (30 trajectories of 2000 steps, data seed 5, as in the paper):
```uv run python generate_dataset.py -m +experiment=pdeparamsweep solver/damping=freqdependent train.num_variations=30 train.num_steps_train=2000 data.seed=5 hydra.sweep.dir=output/generate_dataset```
Runs without `data.extrapolation` in their saved configuration simulate the extrapolation data instead.

Training standardises the data as `(data - data.mean) / data.std` per channel.
`data.mean` defaults to 0 (scale only, as in the paper; the evaluation scripts assume it) and `data.std` to the std of the training split, e.g. `'data.std=[0.04,130.0]'` fixes it.
The paper's runs used the std of the output steps of all trajectories, validation included, so the scale differs slightly unless `data.std` is set.

### test run
```uv run python train_2d_plate.py -m +experiment=testrun solver/damping=freqdependent```

### Train the models
```uv run python train_2d_plate.py -m +experiment=pdeparamsweep solver/damping=freqdependent```

### get validation results and generate tex table
```uv run python validate_2d_plate.py ./output/pdeparamsweep```

### get plots for the extrapolation
The highlighted runs shown in the paper are
gamma = 1.0, kappa = 0.1 run 4
gamma = 1.0, kappa = 1.0 run 4
gamma = 100.0, kappa = 0.1 run 2
```uv run python evaluate_extrapolation.py ./output/pdeparamsweep```

### Spectral layer of FGRU and FRNN
The original `FourierConv2d` keeps only the kx >= 0 half of the `rfft2` output, which drops half of the 2D spectrum.
The results in the paper use this layer, which remains the default (`nnarch.full_spectrum=false`).
Set `nnarch.full_spectrum=true` to keep the full spectrum, e.g.
```uv run python train_2d_plate.py -m +experiment=testrun solver/damping=freqdependent nnarch.full_spectrum=true```

### Training the models separately
Each model has its own optimiser and learning-rate schedule, and its own initialisation seed, so it trains the same alone as alongside the others.
- `train.models` selects the models of a run (default `[gru,rnn,ref]`); e.g. `'train.models=[gru],[rnn],[ref]'` in a multirun gives one job per model, writing to the same run directory.
- `train.clip_jointly` clips the concatenated gradients of the run's models together, as in the paper (`pdeparamsweep` sets it); the default `false` clips each model on its own.
  With the log10 loss, the gradient grows as the loss falls, so under joint clipping the model with the lowest loss shrinks the steps of the others.

### Plate initial conditions from the 2D wave solver
`train.ic=impulse` or `train.ic=impulse_random` start the plate at rest from the initial pressure of the wave2d solver (zero on the boundary, maximum 1), with the random draws of the wave2d dataset: with the same `data.seed`, the plate and the wave2d datasets start from the same fields.
The wave solver starts with zero particle velocity, so this is the same initial state of the wave equation; the trajectories diverge afterwards through the boundary conditions and the dispersion of the finite differences at high wavenumbers.

### Full sweep: spectrum and gradient clipping
`slurm/fullsweep.sbatch` trains 6 datasets x seeds 0–5 x {original, full-spectrum} FourierConv2d x {joint, separate} gradient clipping, as 288 SLURM array tasks, one per line of `slurm/fullsweep_runs.txt`:
- datasets: the 4 (gamma, kappa) of the paper; the plate solver with the parameters of the 2D wave equation of Parker et al. (gamma = 340, kappa = 0, no damping, mixed initial conditions; the boundary conditions differ); the 2D wave equation of Parker et al. itself (`solver=wave2d`: FTM, 25 modes, 3 channels, half impulses and half random initial conditions) with 40 steps;
- joint clipping trains FGRU, FRNN and REF in one task; separate clipping trains one model per task, and the three tasks share the run directory;
- runs go to `$FA2023_OUTPUT/fullspectrumsweep/<data.name>/ic_<ic>/full_spectrum_<bool>-clip_jointly_<bool>/seed_<n>`.

`slurm/make_run_list.py` writes the run list and `slurm/fullsweep_datasets.txt`; it checks that every line fixes every swept setting of `conf/experiment/fullspectrumsweep.yaml`.

From the repository root, after `uv sync --extra cu126`:
```bash
export FA2023_OUTPUT=/path/to/scratch/fa2023              # also seen by the SLURM jobs
bash slurm/generate_datasets.sh                           # 12 datasets, ~8 GB, CPU, ~15 min
sbatch slurm/fullsweep.sbatch                             # all 288 tasks
sbatch -a 1-4 slurm/fullsweep.sbatch train.epochs=1       # quick check of the first run
uv run python validate_2d_plate.py $FA2023_OUTPUT/fullspectrumsweep
uv run python evaluate_extrapolation.py $FA2023_OUTPUT/fullspectrumsweep
```
Arguments after the script name go to `train_2d_plate.py`; `-a 1-288%40` limits the number of simultaneous tasks.
The result tables report the first two channels (displacement and velocity; pressure and x-velocity for the FTM dataset).

### Run the tests
```uv run pytest```
