"""Temporal spectra of the plate gamma=340 (plucks) and FTM wave2d extrapolation datasets.

Writes fig_temporal_spectra.png and fig_wavenumber_frequency.png next to this file and
prints the numbers of the report.

Usage, from the repository root:
    uv run --extra cpu python docs/results/plate-vs-wave2d-spectra/spectra.py $FA2023_OUTPUT/datasets
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.fft import dct, dst, rfft, rfftfreq  # noqa: E402

HERE = Path(__file__).resolve().parent
FS, T, C = 48000, 2000, 340.0
LX, LY = 1.0, 0.95
HX, HY = 1 / 39, 0.95 / 37  # plate grid spacing (40 x 38 points)
FTM_TOP = C / 2 * np.hypot(25 / LX, 25 / LY)  # frequency of the highest FTM mode (Hz)
F = rfftfreq(T, 1 / FS)


def load(datasets):
    plate = np.load(
        f"{datasets}/gamma_340.0-kappa_0.0-damping_nodamping/ic_pluck-n_30-steps_2000-seed_5.npy",
        mmap_mode="r",
    )
    wave = np.load(
        f"{datasets}/wave2d_ftm/ic_impulse_random-n_30-steps_2000-seed_5.npy", mmap_mode="r"
    )
    # name: (data, ICs, {channel label: index}); wave2d: first half impulses, second half random
    return {
        "plate γ=340, plucks": (plate, range(0, 30), {"u": 0, "v": 1}),
        "FTM, impulses": (wave, range(0, 15), {"p": 0, "vx": 1, "vy": 2}),
        "FTM, random": (wave, range(15, 30), {"p": 0, "vx": 1, "vy": 2}),
    }


def psd(data, ics, ch, t0=0, t1=T):
    """Power spectrum averaged over ICs and all grid points (Hann window, steps t0..t1-1).

    Averaging power (not magnitude) over the whole grid gives the energy per mode at its
    frequency, independent of the observation point, because the modes are orthogonal.
    """
    window = np.hanning(t1 - t0)[:, None, None]
    total = 0
    for i in ics:
        x = np.asarray(data[i, t0:t1, :, :, ch], dtype=np.float64)
        x = x - x.mean(axis=0)
        total = total + (np.abs(rfft(x * window, axis=0)) ** 2).mean(axis=(1, 2))
    return total / len(ics)


def quantiles(p, freqs=F, qs=(0.5, 0.9, 0.99)):
    """Frequencies (kHz) below which the fractions qs of the power lie."""
    c = np.cumsum(p) / p.sum()
    return [freqs[np.searchsorted(c, q)] / 1000 for q in qs]


def bands(p, width=200.0):
    """Power summed in constant-width bands: the envelope of the mode lines."""
    edges = np.arange(0, FS / 2 + width, width)
    s = np.bincount(np.digitize(F, edges) - 1, weights=p, minlength=len(edges))
    return (edges[:-1] + edges[1:]) / 2, s[: len(edges) - 1]


def fd_frequency(kx, ky):
    """Angular frequency of the 5-point finite-difference wave equation (exact in time)."""
    return 2 * C * np.sqrt(np.sin(kx * HX / 2) ** 2 / HX**2 + np.sin(ky * HY / 2) ** 2 / HY**2)


def modal_transform(x, basis):
    """Spatial transform of (T, X, Y) fields into the solver's own modes.

    Plate (u = 0 at the walls): DST-I of the interior points, modes sin(m pi x / L).
    FTM (rigid walls): DCT-I of all points, modes cos(m pi x / L).
    Returns the coefficients and the wavenumbers (kx, ky) of the modes.
    """
    if basis == "sine":
        m = dst(dst(x[:, 1:-1, 1:-1], type=1, axis=1), type=1, axis=2)
        mx, my = np.arange(1, m.shape[1] + 1), np.arange(1, m.shape[2] + 1)
    else:
        m = dct(dct(x, type=1, axis=1), type=1, axis=2)
        mx, my = np.arange(m.shape[1]), np.arange(m.shape[2])
    return m, mx * np.pi / LX, my * np.pi / LY


def kf_energy(data, ics, ch, basis, kbins, fbins):
    """Energy over (|k|, f): modal transform in space, Hann-windowed rfft in time."""
    window = np.hanning(T)[:, None, None]
    acc = 0
    for i in ics:
        m, kx, ky = modal_transform(np.asarray(data[i, :, :, :, ch], dtype=np.float64), basis)
        acc = acc + np.abs(rfft(m * window, axis=0)) ** 2
    k = np.hypot(kx[:, None], ky[None, :])
    ki = np.digitize(k, kbins) - 1
    fi = np.digitize(F, fbins) - 1
    ok_f = (fi >= 0) & (fi < len(fbins) - 1)
    energy = np.zeros((len(kbins) - 1, len(fbins) - 1))
    for a in range(k.shape[0]):
        for b in range(k.shape[1]):
            if 0 <= ki[a, b] < len(kbins) - 1:
                np.add.at(energy[ki[a, b]], fi[ok_f], acc[ok_f, a, b])
    return energy


def plate_dispersion_shares(data, ics, ch):
    """Energy shares of the plate modes by finite-difference frequency error."""
    energy = 0
    for i in ics:
        m, kx, ky = modal_transform(np.asarray(data[i, ::10, :, :, ch], dtype=np.float64), "sine")
        energy = energy + (m**2).mean(axis=0)
    energy = energy / energy.sum()
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    w_exact = C * np.hypot(kxg, kyg)
    w_fd = fd_frequency(kxg, kyg)
    error = 1 - w_fd / w_exact
    lag = lambda steps: (w_exact - w_fd) * steps / FS  # noqa: E731
    return {
        "frequency error > 1 %": energy[error > 0.01].sum(),
        "frequency error > 5 %": energy[error > 0.05].sum(),
        "phase lag > pi after 39 steps": energy[lag(39) > np.pi].sum(),
        "phase lag > pi after 2000 steps": energy[lag(2000) > np.pi].sum(),
    }


def figure_temporal(sets, spectra):
    pairs = [
        ("plate γ=340, plucks", "u", "C0", "-"),
        ("plate γ=340, plucks", "v", "C0", ":"),
        ("FTM, impulses", "p", "C1", "-"),
        ("FTM, random", "p", "C2", "-"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, lab, col, ls in pairs:
        fc, s = bands(spectra[(name, lab)])
        axs[0].plot(fc / 1000, 10 * np.log10(s / s.max()), color=col, ls=ls, label=f"{name}: {lab}")
        p = spectra[(name, lab)]
        axs[1].plot(F / 1000, np.cumsum(p) / p.sum(), color=col, ls=ls, label=f"{name}: {lab}")
    axs[0].axvline(FTM_TOP / 1000, color="gray", lw=0.8, ls="--")
    axs[0].text(FTM_TOP / 1000 + 0.1, -57, "highest FTM mode", fontsize=8, color="gray")
    axs[0].set(xlim=(0, 10), ylim=(-60, 3), xlabel="frequency (kHz)",
               ylabel="power in 200 Hz bands (dB re maximum)",
               title="(a) Power spectrum, averaged over points and ICs (u and p obey the same equation)")
    axs[1].set(xlim=(0, 8), xlabel="frequency (kHz)", ylabel="cumulative fraction of channel power",
               title="(b) Cumulative power (linear, as the MSE weights it)")
    top = axs[1].secondary_xaxis(
        "top",
        functions=(lambda k: 2 * np.pi * k * 1000 / FS, lambda r: r * FS / (2 * np.pi * 1000)),
    )
    top.set_xlabel("phase advance per time step (rad)")
    for ax in axs:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_temporal_spectra.png", dpi=100)


def figure_kf(sets):
    kbins = np.linspace(0, 60 * np.pi, 61)
    fbins = np.linspace(0, 10000, 101)
    k = np.linspace(0, 60 * np.pi, 400)
    panels = [
        ("plate γ=340, plucks", 0, "sine", "u"),
        ("FTM, impulses", 0, "cosine", "p"),
        ("FTM, random", 0, "cosine", "p"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, ch, basis, lab) in zip(axs, panels):
        data, ics, _ = sets[name]
        energy = kf_energy(data, ics, ch, basis, kbins, fbins)
        db = 10 * np.log10(energy / energy.max() + 1e-12)
        mesh = ax.pcolormesh(kbins / np.pi, fbins / 1000, db.T, vmin=-60, vmax=0, cmap="magma")
        ax.plot(k / np.pi, C * k / (2 * np.pi) / 1000, "c--", lw=1, label="exact wave: f = c|k| / 2π")
        for label, sx, sy, ls in [("FD scheme, k along x", 1, 0, "w-"),
                                  ("FD scheme, k diagonal", 2**-0.5, 2**-0.5, "w:")]:
            kx, ky = k * sx, k * sy
            ok = (kx <= np.pi / HX) & (ky <= np.pi / HY)
            ax.plot(k[ok] / np.pi, fd_frequency(kx[ok], ky[ok]) / (2 * np.pi) / 1000, ls, lw=1, label=label)
        ax.set(xlabel="|k| / π (1/m)", ylabel="frequency (kHz)", xlim=(0, 60), ylim=(0, 10),
               title=f"{name}: {lab}")
        ax.legend(fontsize=7, loc="upper left")
    fig.colorbar(mesh, ax=axs, label="energy (dB re maximum)", shrink=0.9)
    fig.savefig(HERE / "fig_wavenumber_frequency.png", dpi=100, bbox_inches="tight")


def main(datasets):
    sets = load(datasets)
    spectra = {
        (name, lab): psd(data, ics, ch)
        for name, (data, ics, chans) in sets.items()
        for lab, ch in chans.items()
    }
    print("Frequency (kHz) below which 50 / 90 / 99 % of the channel power lies:")
    for (name, lab), p in spectra.items():
        print(f"  {name:22s} {lab:2s}: " + " / ".join(f"{q:.2f}" for q in quantiles(p)))

    print("Stationarity: 50 / 90 % quantiles (kHz) in steps 0-499 and 1500-1999, power ratio:")
    f500 = rfftfreq(500, 1 / FS)
    for name, lab in [("plate γ=340, plucks", "u"), ("FTM, impulses", "p"), ("FTM, random", "p")]:
        data, ics, chans = sets[name]
        early = psd(data, ics, chans[lab], 0, 500)
        late = psd(data, ics, chans[lab], 1500, 2000)
        qe, ql = quantiles(early, f500, (0.5, 0.9)), quantiles(late, f500, (0.5, 0.9))
        print(f"  {name:22s} {lab}: {qe[0]:.2f} / {qe[1]:.2f} and {ql[0]:.2f} / {ql[1]:.2f}, "
              f"ratio {late.sum() / early.sum():.3f}")

    print("Plate gamma=340: energy share by finite-difference dispersion:")
    data, ics, chans = sets["plate γ=340, plucks"]
    for lab, ch in chans.items():
        shares = plate_dispersion_shares(data, ics, ch)
        print(f"  {lab}: " + "; ".join(f"{key} {value:.3f}" for key, value in shares.items()))

    figure_temporal(sets, spectra)
    figure_kf(sets)


if __name__ == "__main__":
    main(sys.argv[1])
