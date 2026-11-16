# Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers
Code for the paper Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers

## Reproducing the results
GPU needed, might need to adjust batch size depending on GPU RAM

### Create the environment
The project uses [uv](https://docs.astral.sh/uv/); an extra selects the PyTorch backend:
```uv sync --extra cu126```  CUDA 12.6 PyTorch (NVIDIA driver R525 or newer)
```uv sync --extra cpu```  CPU-only PyTorch

`uv run` keeps the backend installed by the last `uv sync`.

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

### Run the tests
```uv run pytest```
