# Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers
Code for the paper Physical Modelling of Stiff Membrane Vibration Using Neural Networks with Spectral Convolution Layers

## Reproducing the results
GPU needed, might need to adjust batch size depending on GPU RAM

### Create the environment
```conda env create -f environment.yml```

### test run
```conda activate platefno && python train_2d_plate.py -m +experiment=testrun solver/damping=freqdependent```

### Train the models
```conda activate platefno && python train_2d_plate.py -m +experiment=pdeparamsweep solver/damping=freqdependent```

### get validation results and generate tex table
```conda activate platefno && python validate_2d_plate.py ./output/pdeparamsweep```

### get plots for the extrapolation
The highlighted runs shown in the paper are
gamma = 1.0, kappa = 0.1 run 4
gamma = 1.0, kappa = 1.0 run 4
gamma = 100.0, kappa = 0.1 run 2
```conda activate platefno && python evaluate_extrapolation.py ./output/pdeparamsweep```
