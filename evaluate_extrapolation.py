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
    rename_model,
)
from platefno.util.plot import plot_mse_per_timestep,plot_run_output_single_ic

# function 

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

    # Solver and domain parameters
    fs = cfg.domain.sampling_rate
    dur = steps / fs
    print(f"Simulation duration: {dur}")
    t60 = get_t60(cfg)

    #######################################################################################################################
    # The solver is used to generate the training data
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=t60,
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )
    # Last dimension is for the 2 channels (displacement, velocity)
    model_input, ground_truth_output = create_dataset(cfg, solver, num_variations=num_variations)
    # Save model input as numpy array
    np.save(os.path.join(dir_name, f"model_input_{steps}.npy"), model_input)
    model_input = torch.from_numpy(model_input)
    ground_truth_output = torch.from_numpy(ground_truth_output)

    # Load saved normalization_multiplier
    normalization_multiplier_saved = get_norms(dir_name)

    model_input *= normalization_multiplier_saved
    ground_truth_output *= normalization_multiplier_saved

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

    # Solver and domain parameters
    fs = cfg.domain.sampling_rate
    dur = steps / fs
    print(f"Simulation duration: {dur}")
    t60 = get_t60(cfg)

    #######################################################################################################################
    # The solver is used to generate the training data
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=t60,
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )
    # Last dimension is for the 2 channels (displacement, velocity)
    model_input, ground_truth_output = create_dataset(cfg, solver, num_variations=num_variations)
    # Save model input as numpy array
    np.save(os.path.join(dir_name, f"model_input_{steps}.npy"), model_input)
    model_input = torch.from_numpy(model_input)
    ground_truth_output = torch.from_numpy(ground_truth_output)

    # Load saved normalization_multiplier
    normalization_multiplier_saved = get_norms(dir_name)

    model_input *= normalization_multiplier_saved
    ground_truth_output *= normalization_multiplier_saved

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

    results = []
    # create a dict with the results
    results.append(
        {
            "model": "gru",
            "gamma": cfg.solver.gamma,
            "kappa": cfg.solver.kappa,
            "ic_train": cfg.train.ic,
            "ic_eval": cfg.train.ic,
            "seed_train": cfg.train.random_seed,
            "seed_eval": seed,
            "div_rate": val_gru_divergence_rate,
        }
    )
    results.append(
        {
            "model": "rnn",
            "gamma": cfg.solver.gamma,
            "kappa": cfg.solver.kappa,
            "ic_train": cfg.train.ic,
            "ic_eval": cfg.train.ic,
            "seed_train": cfg.train.random_seed,
            "seed_eval": seed,
            "div_rate": val_rnn_divergence_rate,
        }
    )
    results.append(
        {
            "model": "ref",
            "gamma": cfg.solver.gamma,
            "kappa": cfg.solver.kappa,
            "ic_train": cfg.train.ic,
            "ic_eval": cfg.train.ic,
            "seed_train": cfg.train.random_seed,
            "seed_eval": seed,
            "div_rate": val_ref_divergence_rate,
        }
    )
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

        # Solver and domain parameters
        fs = cfg.domain.sampling_rate
        dur = steps / fs
        print(f"Simulation duration: {dur}")
        t60 = get_t60(cfg)

        #######################################################################################################################
        # The solver is used to generate the training data
        solver = LinearPlateSolver(
            SR=fs,
            TF=dur,
            gamma=cfg.solver.gamma,
            kappa=cfg.solver.kappa,
            t60=t60,
            aspect_ratio=cfg.domain.aspect_ratio,
            Nx=cfg.domain.nx,
        )
        # Last dimension is for the 2 channels (displacement, velocity)
        model_input, ground_truth_output = create_dataset(cfg, solver, num_variations=num_variations)

        # Print shape of ground truth output
        print(f"Ground truth output shape: {ground_truth_output.shape}")
        
        model_input = torch.from_numpy(model_input)
        ground_truth_output = torch.from_numpy(ground_truth_output)
        # Load saved normalization_multiplier
        normalization_multiplier_saved = get_norms(run_dir)

        model_input *= normalization_multiplier_saved
        ground_truth_output *= normalization_multiplier_saved

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
    # average over seeds
    df_filtered = (
        df.groupby(["model", "gamma", "kappa", "ic_train", "ic_eval"])
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
            "div_rate": "Rate of diverging runs",
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

    # loop over all the runs in the directory
    df_total = pd.DataFrame()
    for run_dir in get_run_dirs(dir_name):
        cfg = get_config(run_dir)
        # only evaluate if the run is for plucks
        if cfg.train.ic == "pluck":
            print(run_dir)
            df_run = evaluate_run_extrapolation_divergence(run_dir, steps=2000, num_variations=30, seed=5)
            df_total = pd.concat(
            [df_total,df_run],
            axis=0,
        )
    df_total.reset_index(drop=True, inplace=True)
    df_total.to_feather(os.path.join(dir_name, "divergence.feather"))
    create_latex_table_div_single_ic(df_total, ic="pluck", file="div_rate.tex")


    # Plot the extrapolation results aggregated for each combination of gamma and kappa
    
    list_cases_dirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]
    for case_dir in list_cases_dirs:
        for highlight_run in [2,4,8]:
            plot_combined_extrapolation(os.path.join(case_dir, "ic_pluck"), steps=2000, num_variations=30, seed=5, highlight_run=(0, highlight_run),linestyles = ["solid", "dashed", "dotted"])

    timer_end = time.time()
    print(f"Time elapsed: {timer_end - timer_start}")
    
        