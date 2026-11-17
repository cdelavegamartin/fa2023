import numpy as np
import torch
import os
import pandas as pd


from platefno.solver.linear_plate_solver import LinearPlateSolver, create_dataset
from platefno.util.conf import get_config, get_t60
from platefno.util.eval import (
    get_run_dirs,
    calculate_mse_per_timestep,
    load_models_from_dir,
    load_data,
    run_model_inference,
    get_norms,
    get_divergence_rate,
    format_parameter,
    label_variants,
    load_extrapolation_data,
    rename_model,
    run_settings,
)
from platefno.nn.models import MODEL_NAMES
from platefno.util.plot import plot_mse_per_timestep,plot_run_output_single_ic

# function 

def extrapolation_data(run_dir, cfg, steps, num_variations):
    """Normalised extrapolation input and target of a run, as torch tensors.

    Reads data.extrapolation.path when the run has it; older runs simulate the
    plate with the numpy seed set by the caller, as before.
    """
    data = load_extrapolation_data(run_dir, steps, num_variations)
    if data is None:
        fs = cfg.domain.sampling_rate
        solver = LinearPlateSolver(
            SR=fs,
            TF=steps / fs,
            gamma=cfg.solver.gamma,
            kappa=cfg.solver.kappa,
            t60=get_t60(cfg),
            aspect_ratio=cfg.domain.aspect_ratio,
            Nx=cfg.domain.nx,
        )
        norms = get_norms(run_dir).numpy()
        data = tuple(
            d * norms for d in create_dataset(cfg, solver, num_variations=num_variations)
        )
    return tuple(torch.from_numpy(np.ascontiguousarray(d)) for d in data)


# function to evaluate a single run but on an arbitrary number of steps
def evaluate_run_extrapolation(dir_name, steps=100, num_variations=10, seed=0):

    debug = False
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    cfg = get_config(dir_name)

    device = torch.device(cfg.train.device)

    # Set seed for reproducibility
    # seed = cfg.train.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_input, ground_truth_output = extrapolation_data(
        dir_name, cfg, steps, num_variations
    )

    # Move data to device
    model_input = model_input.to(device)
    ground_truth_output = ground_truth_output.to(device)

    # Calculate MSE per timestep
    (
        val_gru_mse_per_step,
        val_rnn_mse_per_step,
        val_ref_mse_per_step,
    ) = calculate_mse_per_timestep(
        (model_gru, model_rnn, model_ref),
        model_input,
        ground_truth_output,
    )

    # Calculate divergence rate
    (
        val_gru_divergence_rate,
        val_rnn_divergence_rate,
        val_ref_divergence_rate,
    ) = get_divergence_rate(
        (val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step),)
    

    if debug:
        print(f"GRU divergence rate: {val_gru_divergence_rate}")
        print(f"RNN divergence rate: {val_rnn_divergence_rate}")
        print(f"REF divergence rate: {val_ref_divergence_rate}")


    # Plot the MSE per timestep
    fig, axs = plot_mse_per_timestep(
        (
            val_gru_mse_per_step,
            val_rnn_mse_per_step,
            val_ref_mse_per_step,
        ),
        dir_name, fname=f"extrapolation_mse_per_step_{steps}_oldnorm_{seed}.pdf", average=False, highlight=0, linestyle="dashed", label="extrapolation"
    )

    fig.savefig(
        os.path.join(dir_name, "validation", f"extrapolation_mse_per_step_{steps}_oldnorm_{seed}.pdf"),
        bbox_inches="tight",
    )
    # Save the model input
    torch.save(model_input, os.path.join(dir_name, f"model_input_{steps}.pt"))

    return


# function to evaluate a single run on an arbritrary number of steps and make a table with the divergence rate
def evaluate_run_extrapolation_divergence(dir_name, steps=100, num_variations=10, seed=0):

    debug = True
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    cfg = get_config(dir_name)

    device = torch.device(cfg.train.device)

    # Set seed for reproducibility
    # seed = cfg.train.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_input, ground_truth_output = extrapolation_data(
        dir_name, cfg, steps, num_variations
    )

    # Move data to device
    model_input = model_input.to(device)
    ground_truth_output = ground_truth_output.to(device)

    # Calculate MSE per timestep
    (
        val_gru_mse_per_step,
        val_rnn_mse_per_step,
        val_ref_mse_per_step,
    ) = calculate_mse_per_timestep(
        (model_gru, model_rnn, model_ref),
        model_input,
        ground_truth_output,
    )

    # Calculate divergence rate
    (
        val_gru_divergence_rate,
        val_rnn_divergence_rate,
        val_ref_divergence_rate,
    ) = get_divergence_rate(
        (val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step),)
    

    if debug:
        print(f"GRU divergence rate: {val_gru_divergence_rate}")
        print(f"RNN divergence rate: {val_rnn_divergence_rate}")
        print(f"REF divergence rate: {val_ref_divergence_rate}")

    # One row per model, with the settings that identify the run
    rates = (val_gru_divergence_rate, val_rnn_divergence_rate, val_ref_divergence_rate)
    results = [
        {
            "model": name,
            **run_settings(cfg),
            "ic_eval": cfg.train.ic,
            "seed_eval": seed,
            "div_rate": rate,
        }
        for name, rate in zip(MODEL_NAMES, rates)
    ]
    if len(results) == 0:
        return None
    df = pd.DataFrame.from_dict(results)
    df.to_feather(os.path.join(dir_name, "validation", "div_rate.feather"))
    return df

def plot_combined_extrapolation(dir_name, steps=100, num_variations=10, seed=0, highlight_run=(0,0), linestyles = ["solid", "dashed", "dotted"]):
    fig = None
    axs = None
    linestyles = linestyles
    for run_dir in get_run_dirs(dir_name):
        cfg = get_config(run_dir)
        print(run_dir)        
        # only evaluate if the run is for plucks
        model_gru, model_rnn, model_ref = load_models_from_dir(run_dir)

        device = torch.device(cfg.train.device)

        # Set seed for reproducibility
        model_seed = cfg.train.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        model_input, ground_truth_output = extrapolation_data(
            run_dir, cfg, steps, num_variations
        )

        # Move data to device
        model_input = model_input.to(device)
        ground_truth_output = ground_truth_output.to(device)

        # Calculate MSE per timestep
        (
            val_gru_mse_per_step,
            val_rnn_mse_per_step,
            val_ref_mse_per_step,
        ) = calculate_mse_per_timestep(
            (model_gru, model_rnn, model_ref),
            model_input,
            ground_truth_output,
        )

        if model_seed == highlight_run[0]:
            highlight = highlight_run[1]
        else:
            highlight = None

        if steps > 3*cfg.train.num_steps_train:
            plot_steps = [0, cfg.train.num_steps_train-1, 2*(cfg.train.num_steps_train-1), ground_truth_output.shape[1]-1]
            train_step = cfg.train.num_steps_train-1
        elif steps > cfg.train.num_steps_train:
            plot_steps = [0, cfg.train.num_steps_train-1, ground_truth_output.shape[1]-1]
            train_step = cfg.train.num_steps_train-1
        else:
            plot_steps = [0, steps//2, ground_truth_output.shape[1]-1]

        #  Set the xticks 
        xticks = [0, steps//2]

        val_mse_per_step = (
                val_gru_mse_per_step,
                val_rnn_mse_per_step,
                val_ref_mse_per_step,
            )
        # Plot the MSE per timestep
        fig, axs = plot_mse_per_timestep(
            val_mse_per_step,
            run_dir, 
            fname=f"extrapolation_mse_per_step_{steps}_oldnorm_agg.pdf", 
            average=False, highlight=highlight, linestyle=linestyles[model_seed], label=f"seed={model_seed}", plot_velocity=False, fig=fig, axs=axs, xticks =xticks, train_step=train_step
        )
    fig.savefig(
        os.path.join(dir_name, f"extrapolation_mse_per_step_{steps}_oldnorm_agg_h{highlight}.pdf"),
        bbox_inches="tight",
    )

    

    # If the model seed is 0, take the highlighted IC and plot it
    if highlight is not None:
        output_sequence_gru, output_sequence_rnn, output_sequence_ref = run_model_inference((model_gru, model_rnn, model_ref), model_input[:, 0, ...], ground_truth_output.shape[1])
        fig2, axs2 = plot_run_output_single_ic(ground_truth_output[highlight, ...], (output_sequence_gru[highlight, ...], output_sequence_rnn[highlight, ...], output_sequence_ref[highlight, ...]),plot_steps=plot_steps,fs=fs)
        fig2.savefig(
            os.path.join(dir_name, f"highlighted_ic_plot_steps_{steps}_h{highlight}.pdf"),
            bbox_inches="tight",
        )
    
    return fig, axs


def create_latex_table_div_single_ic(
    df, ic="pluck", file="./crossval_total.tex"
):
    # name each run variant (e.g. full spectrum) so that grouping keeps them apart
    df = label_variants(df)
    # average over seeds
    df_filtered = (
        df.groupby(["dataset", "model", "gamma", "kappa", "ic_train", "ic_eval"], dropna=False)
        .agg({"div_rate": ["mean", "min", "max"]})
        .reset_index()
    )
    # filter for the correct gamma, kappa and model
    df_filtered = df_filtered.loc[
        (df_filtered["ic_train"] == ic) & (df_filtered["ic_eval"] == ic)
    ].reset_index(drop=True)

    df_filtered.drop(columns=["ic_train", "ic_eval"], inplace=True)

    # format the model column
    df_filtered["model"] = df_filtered["model"].apply(rename_model)
    # format the gamma and kappa columns
    df_filtered["gamma"] = df_filtered["gamma"].apply(format_parameter)
    df_filtered["kappa"] = df_filtered["kappa"].apply(format_parameter)
    # Combine gamma and kappa for display
    # df_filtered = combine_gamma_kappa(df_filtered)
    #  change the order of the columns
    df_filtered.sort_values(by=["model"], inplace=True)
    df_filtered.sort_values(by=["kappa"], inplace=True)
    df_filtered.sort_values(by=["gamma"], inplace=True)
    df_filtered.sort_values(by=["dataset"], inplace=True, kind="stable")

    # df_filtered.set_index(["gamma_kappa", "model"], inplace=True)
    df_filtered.set_index(["dataset", "gamma", "kappa", "model"], inplace=True)
    # rename columns
    df_filtered.rename(
        columns={
            "div_rate": "Rate of diverging runs",
        },
        inplace=True,
    )
    df_filtered.rename_axis(["Dataset", "$\gamma$", "$\kappa$", "Model"], inplace=True)

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
        float_format="{:.4f}".format,
        index=True,
        column_format="lccccc",
        caption=caption,
        label=f"tab:extrap_div_{exp_dir}_ic-{ic}",
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

    # Divergence of every run over the extrapolation length of the paper
    run_dirs = get_run_dirs(dir_name)
    df_total = pd.concat(
        [
            evaluate_run_extrapolation_divergence(
                run_dir, steps=2000, num_variations=30, seed=5
            )
            for run_dir in run_dirs
        ],
        axis=0,
    )
    df_total.reset_index(drop=True, inplace=True)
    df_total.to_feather(os.path.join(dir_name, "divergence.feather"))
    for ic in df_total["ic_train"].unique():
        create_latex_table_div_single_ic(
            df_total, ic=ic, file=os.path.join(dir_name, f"div_rate_{ic}.tex")
        )

    # Extrapolation MSE of the runs of each directory that holds seed_<n> runs
    # (a (gamma, kappa) case in pdeparamsweep, a run variant in fullspectrumsweep)
    for group_dir in sorted({os.path.dirname(run_dir) for run_dir in run_dirs}):
        for highlight_run in [2, 4, 8]:
            plot_combined_extrapolation(
                group_dir,
                steps=2000,
                num_variations=30,
                seed=5,
                highlight_run=(0, highlight_run),
                linestyles=["solid", "dashed", "dotted"],
            )

    timer_end = time.time()
    print(f"Time elapsed: {timer_end - timer_start}")
    
        