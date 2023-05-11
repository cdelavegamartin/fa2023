import os
import torch
import numpy as np
import pathlib
import pandas as pd
from platefno.util.conf import get_config, get_t60
from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.nn.fno_rnn import FNO_RNN_2d
from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d


def get_run_dirs(dir_name):
    run_dirs = []
    # loop over all the subdirectories recursevily in the directory
    for subdir, dirs, files in os.walk(dir_name, topdown=True):
        # if the directory contains a tensorboard log file startin with "events.out.tfevents"
        if any(".hydra" in s for s in dirs):
            run_dirs.append(subdir)
    return run_dirs


def get_val_data_path(dir_name, ic_name="pluck", seed=0):
    # Find the path segment that starts with ic_
    # This is only valid for the pdeparamsweep experiments
    run_dir_parts = pathlib.Path(dir_name).parts
    for i, part in enumerate(run_dir_parts):
        if part.startswith("ic_"):
            # Replace the part with ic_mix
            index = i
            val_data_path = (
                pathlib.Path(*run_dir_parts[:index])
                .joinpath(f"ic_{ic_name}")
                .joinpath(*run_dir_parts[index + 1 : -1])
                .joinpath(f"seed_{seed}")
            )
            return val_data_path
    return None




def load_models_from_dir(dir_name):
    cfg = get_config(dir_name)
    cfg_hydra = get_config(dir_name, config_name="hydra")
    # get output dir from config

    output_dir = dir_name
    num_steps_train = cfg.train.num_steps_train

    # Set torch device if posible if not cpu
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)

    # Solver and domain parameters
    fs = cfg.domain.sampling_rate
    t60 = get_t60(cfg)
    dur = num_steps_train / fs

    # print(f"Simulation duration: {dur}")
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=t60,
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )
    # Instantiate the models
    # print(f"Instantiate GRU model")
    model_gru = torch.nn.DataParallel(
        FNO_GRU_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate RNN model")
    model_rnn = torch.nn.DataParallel(
        FNO_RNN_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate Ref model")
    model_ref = torch.nn.DataParallel(
        FNO_Markov_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)

    # Load the models
    # print(f"Load GRU model")
    model_gru.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_gru.pth"))
    )
    # print(f"Load RNN model")
    model_rnn.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_rnn.pth"))
    )
    # print(f"Load Ref model")
    model_ref.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_ref.pth"))
    )
    return model_gru, model_rnn, model_ref


def load_data(dir_name):
    # Load the model input from the npy file, and convert it to a tensor
    validation_input = torch.from_numpy(
        np.load(os.path.join(dir_name, "validation", "validation_input.npy"))
    )
    # Load ground truth from the npy file, and convert it to a tensor
    # print(f"Load ground truth")
    validation_output = torch.from_numpy(
        np.load(os.path.join(dir_name, "validation", "validation_output.npy"))
    )
    return validation_input, validation_output


def run_model_inference(models, model_input, num_steps):
    # Unpack the models
    model_gru, model_rnn, model_ref = models
    # Put models in evaluation mode
    model_gru.eval()
    model_rnn.eval()
    model_ref.eval()

    with torch.no_grad():
        # Run the models
        # print(f"Run GRU model")
        output_sequence_gru = model_gru(model_input, num_steps)
        # print(f"Run RNN model")
        output_sequence_rnn = model_rnn(model_input, num_steps)
        # print(f"Run Ref model")
        output_sequence_ref = model_ref(model_input, num_steps)
    return (output_sequence_gru, output_sequence_rnn, output_sequence_ref)


def calculate_mse_per_timestep(models, validation_input, validation_output):
    output_sequence_gru, output_sequence_rnn, output_sequence_ref = run_model_inference(
        models, validation_input[:, 0, ...], validation_output.shape[1]
    )

    # Calculate MSE per timestep, per IC, these are tensors of shape (num_ICs, num_timesteps, num_state_variables)
    val_gru_mse = (
        torch.mean((output_sequence_gru - validation_output) ** 2, dim=(2, 3))
        .detach()
        .cpu()
        .numpy()
    )
    val_rnn_mse = (
        torch.mean((output_sequence_rnn - validation_output) ** 2, dim=(2, 3))
        .detach()
        .cpu()
        .numpy()
    )
    val_ref_mse = (
        torch.mean((output_sequence_ref - validation_output) ** 2, dim=(2, 3))
        .detach()
        .cpu()
        .numpy()
    )

    return (val_gru_mse, val_rnn_mse, val_ref_mse)


def calculate_mse(models, validation_input, validation_output):
    output_sequence_gru, output_sequence_rnn, output_sequence_ref = run_model_inference(
        models, validation_input[:, 0, ...], validation_output.shape[1]
    )

    # Calculate MSE these are tensors of shape (num_state_variables,)
    val_gru_mse = (
        torch.mean((output_sequence_gru - validation_output) ** 2, dim=(0, 1, 2, 3))
        .detach()
        .cpu()
        .numpy()
    )
    val_rnn_mse = (
        torch.mean((output_sequence_rnn - validation_output) ** 2, dim=(0, 1, 2, 3))
        .detach()
        .cpu()
        .numpy()
    )
    val_ref_mse = (
        torch.mean((output_sequence_ref - validation_output) ** 2, dim=(0, 1, 2, 3))
        .detach()
        .cpu()
        .numpy()
    )

    return (val_gru_mse, val_rnn_mse, val_ref_mse)


def calculate_mse_crossval(dir_name, ic_eval=["pluck", "random"], num_seeds=3):
    cfg = get_config(dir_name)
    output_dir = dir_name
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    # Set torch device if posible if not cpu
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    results = []
    for ic_name in ic_eval:
        for seed in range(num_seeds):
            # get the validation data
            val_data_dir = get_val_data_path(dir_name, ic_name=ic_name, seed=seed)
            if val_data_dir is None:
                continue
            if not os.path.exists(val_data_dir):
                continue
            validation_input, validation_output = load_data(val_data_dir)
            validation_input = validation_input.to(device)
            validation_output = validation_output.to(device)

            ###############################################
            # The following is a hack to remove a broken run from the dataset when evaluating the models
            # The specific run was identified with the plot_broken.py script
            val_model_broken = pathlib.Path(
                "/home/carlos/projects/platefno/output/pdeparamsweep-full/gamma_1.0-kappa_1.0/ic_pluck/seed_2"
            )
            val_data_broken = pathlib.Path(
                "/home/carlos/projects/platefno/output/pdeparamsweep-full/gamma_1.0-kappa_1.0/ic_pluck/seed_1"
            )
            broken_ic = 85
            if (
                pathlib.Path(val_data_dir) == val_data_broken
                and pathlib.Path(dir_name) == val_model_broken
            ):
                print("PING")
                # Remove the broken run from the dataset
                validation_input = torch.cat(
                    [
                        validation_input[:broken_ic, ...],
                        validation_input[broken_ic + 1 :, ...],
                    ]
                )
                validation_output = torch.cat(
                    [
                        validation_output[:broken_ic, ...],
                        validation_output[broken_ic + 1 :, ...],
                    ]
                )
            ###############################################

            # Calculate MSE for each state variable
            val_gru_mse, val_rnn_mse, val_ref_mse = calculate_mse(
                (model_gru, model_rnn, model_ref),
                validation_input,
                validation_output,
            )

            # create a dict with the results
            results.append(
                {
                    "model": "gru",
                    "gamma": cfg.solver.gamma,
                    "kappa": cfg.solver.kappa,
                    "ic_train": cfg.train.ic,
                    "ic_eval": ic_name,
                    "seed_train": cfg.train.random_seed,
                    "seed_eval": seed,
                    "mse_u": val_gru_mse[0],
                    "mse_v": val_gru_mse[1],
                }
            )
            results.append(
                {
                    "model": "rnn",
                    "gamma": cfg.solver.gamma,
                    "kappa": cfg.solver.kappa,
                    "ic_train": cfg.train.ic,
                    "ic_eval": ic_name,
                    "seed_train": cfg.train.random_seed,
                    "seed_eval": seed,
                    "mse_u": val_rnn_mse[0],
                    "mse_v": val_rnn_mse[1],
                }
            )
            results.append(
                {
                    "model": "ref",
                    "gamma": cfg.solver.gamma,
                    "kappa": cfg.solver.kappa,
                    "ic_train": cfg.train.ic,
                    "ic_eval": ic_name,
                    "seed_train": cfg.train.random_seed,
                    "seed_eval": seed,
                    "mse_u": val_ref_mse[0],
                    "mse_v": val_ref_mse[1],
                }
            )
    if len(results) == 0:
        return None
    df = pd.DataFrame.from_dict(results)
    df.to_feather(os.path.join(output_dir, "validation", "crossval.feather"))
    return df
