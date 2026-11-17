import pandas as pd
from omegaconf import OmegaConf

from platefno.util.eval import label_variants, run_settings
from validate_2d_plate import create_latex_table_val_results_single_ic


def config(nnarch=None, train=None):
    return OmegaConf.create(
        {
            "solver": {"gamma": 100.0, "kappa": 0.1, "damping": {"name": "freqdependent"}},
            "nnarch": {"width": 8, **(nnarch or {})},
            "train": {"ic": "pluck", "random_seed": 0, **(train or {})},
        }
    )


def test_run_settings_defaults_for_old_configs():
    settings = run_settings(config())
    assert settings["full_spectrum"] is False
    assert settings["clip_jointly"] is True


def test_run_settings_reads_new_options():
    settings = run_settings(
        config({"full_spectrum": True}, {"clip_jointly": False})
    )
    assert settings["full_spectrum"] is True
    assert settings["clip_jointly"] is False


def rows(full_spectrum, clip_jointly=False, mse=1.0):
    return [
        {
            "model": model,
            "gamma": 100.0,
            "kappa": 0.1,
            "ic_train": "pluck",
            "ic_eval": "pluck",
            "seed_train": 0,
            "seed_eval": 0,
            "full_spectrum": full_spectrum,
            "clip_jointly": clip_jointly,
            "mse_u": mse,
            "mse_v": mse,
        }
        for model in ("gru", "rnn", "ref")
    ]


def test_labels_name_full_spectrum_and_mixed_clipping():
    df = pd.DataFrame(rows(False) + rows(True))
    assert set(label_variants(df)["model"]) == {
        "FGRU",
        "FRNN",
        "REF",
        "FGRU, full spectrum",
        "FRNN, full spectrum",
        "REF, full spectrum",
    }
    mixed = pd.DataFrame(rows(False, clip_jointly=True) + rows(False, clip_jointly=False))
    assert "FGRU, joint clip" in set(label_variants(mixed)["model"])


def test_labels_for_old_results_without_variant_columns():
    df = pd.DataFrame(rows(False)).drop(columns=["full_spectrum", "clip_jointly"])
    assert set(label_variants(df)["model"]) == {"FGRU", "FRNN", "REF"}


def test_table_keeps_variants_apart(tmp_path):
    df = pd.DataFrame(rows(False, mse=0.5) + rows(True, mse=0.1))
    table = create_latex_table_val_results_single_ic(
        df, ic="pluck", file=str(tmp_path / "val.tex")
    )
    assert len(table) == 6
    models = table.index.get_level_values("Model")
    assert table.loc[("100.0", "0.1", "FGRU")]["Displacement"].iloc[0].startswith("0.5000")
    assert "FGRU, full spectrum" in models
    assert (tmp_path / "val.tex").exists()
