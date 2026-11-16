# Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers
Code for the paper Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers

## Reproducing the results
GPU needed, might need to adjust batch size depending on GPU RAM

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

### Full-spectrum sweep
Every (gamma, kappa) of the paper with the original and the full-spectrum layer (seed 0); both use the same dataset.
```uv run python generate_dataset.py -m +experiment=fullspectrumsweep solver/damping=freqdependent nnarch.full_spectrum=false hydra.sweep.dir=output/generate_dataset```
```uv run python train_2d_plate.py -m +experiment=fullspectrumsweep solver/damping=freqdependent```

### Run the tests
```uv run pytest```
