import os
import torch
import numpy as np
import pathlib
import pandas as pd
from platefno.util.conf import get_config
from omegaconf import OmegaConf

from platefno.data import grid_size
from platefno.nn.models import MODEL_NAMES, build_models


def get_device(cfg):
    """cfg.train.device when it is CUDA and CUDA is available, otherwise the CPU."""
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


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


def read_feather(dir_name, filename="crossval.feather"):
    df = pd.read_feather(os.path.join(dir_name, filename))
    return df


def get_norms(dir_name):
    norms = torch.load(os.path.join(dir_name, "model", "norms.pt"))
    return norms


def load_models_from_dir(dir_name):
    cfg = get_config(dir_name)
    cfg_hydra = get_config(dir_name, config_name="hydra")
    # get output dir from config

    output_dir = dir_name

    device = get_device(cfg)

    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)

    size_x, size_y = grid_size(cfg)

    # Instantiate the models
    model_gru, model_rnn, model_ref = build_models(
        cfg, size_x, size_y, device
    ).values()

    # Load the weights onto the evaluation device, also when they were saved on another
    for name, model in zip(("gru", "rnn", "ref"), (model_gru, model_rnn, model_ref)):
        path = os.path.join(output_dir, "model", f"model_{name}.pth")
        model.load_state_dict(torch.load(path, map_location=device))
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


def load_extrapolation_data(dir_name, steps, num_variations, base_dir="."):
    """Raw extrapolation input (n, 1, x, y, c) and target (n, steps - 1, x, y, c) of
    a run, scaled by the run's normalisation, as numpy arrays.

    Reads the first num_variations trajectories and steps time steps of
    data.extrapolation.path (relative to base_dir); returns None for runs saved
    before that setting existed, which simulate their data instead.
    """
    extrapolation = (get_config(dir_name).get("data") or {}).get("extrapolation")
    if extrapolation is None:
        return None
    path = os.path.join(base_dir, extrapolation.path)
    data = np.load(path, mmap_mode="r")
    if steps > data.shape[1] or num_variations > data.shape[0]:
        raise ValueError(
            f"{path} has {data.shape[0]} trajectories of {data.shape[1]} steps; "
            f"{num_variations} of {steps} were requested"
        )
    data = np.array(data[:num_variations, :steps]) * get_norms(dir_name).numpy()
    return data[:, :1].astype(np.float32), data[:, 1:].astype(np.float32)


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
    """MSE per IC, time step and state variable, (num_ICs, num_timesteps,
    num_state_variables) numpy arrays, one per model.

    Runs one model at a time, so only one output sequence is in memory at once.
    """
    results = []
    with torch.no_grad():
        for model in models:
            model.eval()
            output_sequence = model(validation_input[:, 0, ...], validation_output.shape[1])
            mse = torch.mean((output_sequence - validation_output) ** 2, dim=(2, 3))
            results.append(mse.cpu().numpy())
            del output_sequence
    return tuple(results)


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


# Function to get the failure rate of the model
def get_divergence_rate(mse_per_timestep):
    """Calculate the divergence rate of the model, defined as the fraction of ICs that diverge to infinity"""
    debug = True
    # Unpack the MSE per timestep
    val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step = mse_per_timestep

    if debug:
        # Print the maximum MSE for each model
        print(f"GRU max MSE (u): {np.nanmax(val_gru_mse_per_step[..., 0], axis=1)}")
        print(f"RNN max MSE (u): {np.nanmax(val_rnn_mse_per_step[..., 0], axis=1)}")
        print(f"REF max MSE (u): {np.nanmax(val_ref_mse_per_step[..., 0], axis=1)}")

        print(f"GRU max MSE (v): {np.nanmax(val_gru_mse_per_step[..., 1], axis=1)}")
        print(f"RNN max MSE (v): {np.nanmax(val_rnn_mse_per_step[..., 1], axis=1)}")
        print(f"REF max MSE (v): {np.nanmax(val_ref_mse_per_step[..., 1], axis=1)}")
    gru_div_rate = (
        np.sum(np.nanmax(val_gru_mse_per_step, axis=1) == np.inf)
        / val_gru_mse_per_step.shape[0]
        / val_gru_mse_per_step.shape[2]
    )
    rnn_div_rate = (
        np.sum(np.nanmax(val_rnn_mse_per_step, axis=1) == np.inf)
        / val_rnn_mse_per_step.shape[0]
        / val_rnn_mse_per_step.shape[2]
    )
    ref_div_rate = (
        np.sum(np.nanmax(val_ref_mse_per_step, axis=1) == np.inf)
        / val_ref_mse_per_step.shape[0]
        / val_ref_mse_per_step.shape[2]
    )

    return (gru_div_rate, rnn_div_rate, ref_div_rate)


def run_settings(cfg):
    """Settings that identify a trained run, for the result tables.

    Configurations saved before full_spectrum and clip_jointly existed used the
    original layer and joint clipping.
    """
    return {
        "dataset": OmegaConf.select(cfg, "data.name")
        or default_dataset_name(cfg.solver),
        # None for solvers without these parameters (wave2d)
        "gamma": OmegaConf.select(cfg, "solver.gamma"),
        "kappa": OmegaConf.select(cfg, "solver.kappa"),
        "damping": OmegaConf.select(cfg, "solver.damping.name"),
        "ic_train": cfg.train.ic,
        "seed_train": cfg.train.random_seed,
        "full_spectrum": bool(cfg.nnarch.get("full_spectrum", False)),
        "clip_jointly": bool(cfg.train.get("clip_jointly", True)),
    }


def default_dataset_name(solver):
    """Dataset name of runs saved before data.name existed (gamma and kappa only)."""
    return f"gamma_{solver['gamma']}-kappa_{solver['kappa']}"


VARIANT_DEFAULTS = {"full_spectrum": False, "clip_jointly": True}


def with_variant_columns(df):
    """Copy of df with the dataset and run-variant columns, filled with the
    defaults of run_settings for rows saved before these settings existed."""
    df = df.copy()
    for column, default in VARIANT_DEFAULTS.items():
        if column not in df:
            df[column] = default
        df[column] = df[column].fillna(default).astype(bool)
    if "dataset" not in df:
        df["dataset"] = None
    missing = df["dataset"].isna()
    if missing.any():
        df.loc[missing, "dataset"] = df[missing].apply(default_dataset_name, axis=1)
    return df


def label_variants(df):
    """Copy of df whose "model" column names the model and its run variant.

    E.g. "FGRU, full spectrum"; the clipping mode is named only when df holds
    both modes. Grouping by "model" then keeps the variants apart.
    """
    df = with_variant_columns(df)
    show_clip = df["clip_jointly"].nunique() > 1

    def label(row):
        parts = [rename_model(row["model"])]
        if row["full_spectrum"]:
            parts.append("full spectrum")
        if show_clip:
            parts.append("joint clip" if row["clip_jointly"] else "separate clip")
        return ", ".join(parts)

    df["model"] = df.apply(label, axis=1)
    return df


def validation_seeds(dir_name, ic_name, num_seeds=None):
    """Seeds whose validation data crossval evaluates for an initial condition.

    num_seeds=None takes every seed_<n> directory next to the matching run.
    """
    if num_seeds is not None:
        return range(num_seeds)
    first = get_val_data_path(dir_name, ic_name=ic_name, seed=0)
    if first is None or not first.parent.is_dir():
        return []
    return sorted(
        int(p.name.removeprefix("seed_"))
        for p in first.parent.glob("seed_*")
        if p.name.removeprefix("seed_").isdigit()
    )


def dataset_path(dir_name):
    """data.path of a run, or None for runs saved before it existed."""
    data = get_config(dir_name).get("data")
    return None if data is None else data.get("path")


def repeats_own_dataset(dir_name, other_dir):
    """True when other_dir is another run trained on the same dataset file, so its
    validation data repeats the run's own."""
    if pathlib.Path(other_dir).resolve() == pathlib.Path(dir_name).resolve():
        return False
    if not (pathlib.Path(other_dir) / ".hydra").is_dir():
        return False
    own = dataset_path(dir_name)
    return own is not None and own == dataset_path(other_dir)


def calculate_mse_crossval(dir_name, ic_eval=None, num_seeds=None):
    """MSE of the run's models on the validation data of every seed (and initial
    condition) of the same experiment, once per distinct dataset.

    ic_eval=None evaluates the run's own initial condition, plus pluck and random.
    Runs of other seeds that share the run's dataset are skipped, since their
    validation data is the run's own.
    """
    cfg = get_config(dir_name)
    if ic_eval is None:
        ic_eval = list(dict.fromkeys([cfg.train.ic, "pluck", "random"]))
    output_dir = dir_name
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    normalization_multiplier_model = get_norms(dir_name)
    device = get_device(cfg)

    results = []
    for ic_name in ic_eval:
        for seed in validation_seeds(dir_name, ic_name, num_seeds):
            # get the validation data
            val_data_dir = get_val_data_path(dir_name, ic_name=ic_name, seed=seed)
            if val_data_dir is None:
                continue
            if repeats_own_dataset(dir_name, val_data_dir):
                continue
            if not os.path.exists(val_data_dir):
                continue

            # Get val_data_dir normalizations
            normalization_multiplier_data = get_norms(val_data_dir)

            validation_input, validation_output = load_data(val_data_dir)

            # Normalize the validation data with the normalization from the training data
            validation_input /= normalization_multiplier_data
            validation_output /= normalization_multiplier_data
            validation_input *= normalization_multiplier_model
            validation_output *= normalization_multiplier_model

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

            # Put the validation data on the device
            validation_input = validation_input.to(device)
            validation_output = validation_output.to(device)

            # Calculate MSE for each state variable
            val_gru_mse, val_rnn_mse, val_ref_mse = calculate_mse(
                (model_gru, model_rnn, model_ref),
                validation_input,
                validation_output,
            )

            # One row per model, with the settings that identify the run
            for name, mse in zip(MODEL_NAMES, (val_gru_mse, val_rnn_mse, val_ref_mse)):
                results.append(
                    {
                        "model": name,
                        **run_settings(cfg),
                        "ic_eval": ic_name,
                        "seed_eval": seed,
                        "mse_u": mse[0],
                        "mse_v": mse[1],
                    }
                )
    if len(results) == 0:
        return None
    df = pd.DataFrame.from_dict(results)
    df.to_feather(os.path.join(output_dir, "validation", "crossval.feather"))
    return df


def rename_model(model_name):
    if model_name == "gru":
        return "FGRU"
    elif model_name == "rnn":
        return "FRNN"
    elif model_name == "ref":
        return "REF"
    else:
        return model_name


def format_parameter(x):
    """PDE parameter for the tables; "-" when the run's solver does not have it."""
    return "-" if x is None or pd.isna(x) else f"{x:.1f}"


def format_mse(x, tol=5.0):
    if x < tol and not np.isnan(x):
        return "{:.4f}".format(x)
    else:
        return " - "


# Function to create new column in the dataframe with the mean of the mse, and std in parenthesis
def create_mean_std(df):
    df["mean_std_u"] = (
        df["mse_u"]["mean"].apply(format_mse).astype("str")
        + " ("
        + df["mse_u"]["std"].apply(format_mse).astype("str")
        + ")"
    )

    df["mean_std_v"] = (
        df["mse_v"]["mean"].apply(format_mse).astype("str")
        + " ("
        + df["mse_v"]["std"].apply(format_mse).astype("str")
        + ")"
    )
    return df


# Function to combine gamma and kappa in one column for display purposes
def combine_gamma_kappa(df):
    df["gamma_kappa"] = (
        df["gamma"].apply(lambda x: "$\gamma={:.1f}$".format(x)).astype("str")
        + ", "
        + df["kappa"].apply(lambda x: "$\kappa={:.1f}$".format(x)).astype("str")
    )
    return df
