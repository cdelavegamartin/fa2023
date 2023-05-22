import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import os
import pathlib
import pandas as pd


from platefno.util.conf import get_config
from platefno.util.eval import (
    get_run_dirs,
    read_feather,
    calculate_mse_per_timestep,
    load_models_from_dir,
    load_data,
    calculate_mse_crossval,
    rename_model,
    format_mse,
    create_mean_std,
    combine_gamma_kappa,
)
from platefno.util.plot import plot_mse_per_timestep


def evaluate_run(dir_name):
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    cfg = get_config(dir_name)

    # Set torch device if posible if not cpu
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    validation_input, validation_output = load_data(dir_name)
    validation_input = validation_input.to(device)
    validation_output = validation_output.to(device)

    # Calculate MSE per step
    (
        val_gru_mse_per_step,
        val_rnn_mse_per_step,
        val_ref_mse_per_step,
    ) = calculate_mse_per_timestep(
        (model_gru, model_rnn, model_ref),
        validation_input,
        validation_output,
    )

    plot_mse_per_timestep(
        (
            val_gru_mse_per_step,
            val_rnn_mse_per_step,
            val_ref_mse_per_step,
        ),
        dir_name,
    )
    calculate_mse_crossval(dir_name)

    return


# This function takes in a directory and aggregates the results from all the runs in that directory
def aggregate_experiment_val_results(exp_dir):
    # Get the names of all the runs in the directory
    run_dirs = get_run_dirs(exp_dir)
    df = pd.DataFrame()
    for run_dir in run_dirs:
        df = pd.concat(
            [
                df,
                read_feather(
                    os.path.join(run_dir, "validation"), filename="crossval.feather"
                ),
            ],
            axis=0,
        )
    df.reset_index(drop=True, inplace=True)
    df.to_feather(os.path.join(exp_dir, "crossval_total.feather"))
    return df


# Function to create a latex table from the aggregated results. Mean and std are calculated over the seeds
def create_latex_table_val_results(
    df, gamma=100.0, kappa=0.1, model="gru", file="./crossval_total.tex"
):
    # average over seeds
    df_filtered = (
        df.groupby(["model", "gamma", "kappa", "ic_train", "ic_eval"])
        .agg({"mse_u": ["mean", "std"], "mse_v": ["mean", "std"]})
        .reset_index()
    )
    # filter for the correct gamma, kappa and model
    df_filtered = df_filtered.loc[
        (df_filtered["gamma"] == gamma)
        & (df_filtered["kappa"] == kappa)
        & (df_filtered["model"] == model)
    ].reset_index(drop=True)
    # rename columns
    df_filtered.rename(
        columns={
            "mse_u": "MSE (u)",
            "mse_v": "MSE (v)",
            "ic_train": "training",
            "ic_eval": "evaluation",
        },
        inplace=True,
    )
    df_filtered.drop(columns=["gamma", "kappa", "model"], inplace=True)
    # print(df_filtered)
    df_filtered.to_latex(file, float_format="{:.4f}".format, index=False)
    return df_filtered


def create_latex_table_val_results_alt(
    df, gamma=100.0, kappa=0.1, ic_eval="pluck", file="./crossval_total.tex"
):
    # average over seeds
    df_filtered = (
        df.groupby(["model", "gamma", "kappa", "ic_train", "ic_eval"])
        .agg({"mse_u": ["mean", "std"], "mse_v": ["mean", "std"]})
        .reset_index()
    )
    # filter for the correct gamma, kappa and model
    df_filtered = df_filtered.loc[
        (df_filtered["gamma"] == gamma)
        & (df_filtered["kappa"] == kappa)
        & (df_filtered["ic_eval"] == ic_eval)
    ].reset_index(drop=True)

    df_filtered = create_mean_std(df_filtered)

    # rename columns
    df_filtered.rename(
        columns={
            "mean_std_u": "Displacement",
            "mean_std_v": "Velocity",
            "ic_train": "Train",
        },
        inplace=True,
    )
    df_filtered.drop(
        columns=["gamma", "kappa", "ic_eval", "mse_u", "mse_v"], inplace=True
    )

    # format the model column
    df_filtered["model"] = df_filtered["model"].apply(rename_model)
    #  change the order of the columns
    df_filtered = df_filtered[
        ["Train", "model", "Displacement", "Velocity"]
    ].reset_index(drop=True)
    df_filtered.sort_values(by=["model"], inplace=True)
    df_filtered.sort_values(by=["Train"], inplace=True)
    df_filtered.columns = df_filtered.columns.get_level_values(0)
    df_filtered.set_index(["Train", "model"], inplace=True)
    # print(df_filtered)
    caption = (
        f"MSE \\textit{{mean (std)}}, evaluated on a \\textit{{{ic_eval}}} IC dataset. "
        + f"PDE parameters are $\gamma={gamma}$ and $\kappa={kappa}$. "
        + f"Diverging results (MSE $>$ 5.0) are omitted."
    )
    # This is a hack to get the experiment directory name
    # Wont work in Windows
    exp_dir = os.path.dirname(file).split("/")[-1]
    df_filtered.to_latex(
        file,
        # columns=["training", "model", "Displacement", "Velocity"],
        float_format="{:.4f}".format,
        index=True,
        column_format="lccccc",
        formatters={"model": str.upper},
        caption=caption,
        label=f"tab:crossval_{exp_dir}_eval-{ic_eval}_gamma-{gamma}_kappa-{kappa}",
    )
    return df_filtered


#  create a latex table from a pandas dataframe using only the the runs trained and evaluated on the same IC
#  aggregate all the gammas and kappas together
def create_latex_table_val_results_single_ic(
    df, ic="pluck", file="./crossval_total.tex"
):
    # average over seeds
    df_filtered = (
        df.groupby(["model", "gamma", "kappa", "ic_train", "ic_eval"])
        .agg({"mse_u": ["mean", "std"], "mse_v": ["mean", "std"]})
        .reset_index()
    )
    # filter for the correct gamma, kappa and model
    df_filtered = df_filtered.loc[
        (df_filtered["ic_train"] == ic) & (df_filtered["ic_eval"] == ic)
    ].reset_index(drop=True)

    df_filtered = create_mean_std(df_filtered)

    df_filtered.drop(columns=["ic_train", "ic_eval", "mse_u", "mse_v"], inplace=True)

    # format the model column
    df_filtered["model"] = df_filtered["model"].apply(rename_model)
    # format the gamma and kappa columns
    df_filtered["gamma"] = df_filtered["gamma"].apply(lambda x: f"{x:.1f}")
    df_filtered["kappa"] = df_filtered["kappa"].apply(lambda x: f"{x:.1f}")
    # Combine gamma and kappa for display
    # df_filtered = combine_gamma_kappa(df_filtered)
    #  change the order of the columns
    df_filtered.sort_values(by=["model"], inplace=True)
    df_filtered.sort_values(by=["kappa"], inplace=True)
    df_filtered.sort_values(by=["gamma"], inplace=True)

    # df_filtered.set_index(["gamma_kappa", "model"], inplace=True)
    df_filtered.set_index(["gamma", "kappa", "model"], inplace=True)
    # rename columns
    df_filtered.rename(
        columns={
            "mean_std_u": "Displacement",
            "mean_std_v": "Velocity",
        },
        inplace=True,
    )
    df_filtered.rename_axis(["$\gamma$", "$\kappa$", "Model"], inplace=True)

    # df_filtered.drop(columns=["gamma", "kappa"], inplace=True)
    print(df_filtered)
    caption = (
        f"Results MSE \\textit{{mean (std)}}. "
        + f"Diverging results (MSE $>$ 5.0) are omitted."
    )
    # This is a hack to get the experiment directory name
    # Wont work in Windows
    exp_dir = os.path.dirname(file).split("/")[-1]
    df_filtered.to_latex(
        file,
        # columns=["training", "model", "Displacement", "Velocity"],
        float_format="{:.1f}".format,
        index=True,
        column_format="lccccc",
        caption=caption,
        label=f"tab:val_{exp_dir}_ic-{ic}",
    )
    return df_filtered


if __name__ == "__main__":
    import sys
    import time

    # get the directory name from the command line
    dir_name = sys.argv[1]
    # evaluate_run(dir_name)
    # time how long it takes
    timer_start = time.time()
    print(len(get_run_dirs(dir_name)))

    # loop over all the runs in the directory
    for run_dir in get_run_dirs(dir_name):
        print(run_dir)
        evaluate_run(run_dir)

    df = aggregate_experiment_val_results(dir_name)
    ic = "pluck"
    create_latex_table_val_results_single_ic(
        df,
        ic=ic,
        file=os.path.join(dir_name, f"val_{ic}.tex"),
    )

    timer_end = time.time()
    print(f"Elapsed time: {timer_end - timer_start} seconds")
