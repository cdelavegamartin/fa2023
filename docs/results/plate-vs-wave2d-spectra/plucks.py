"""Plucks in four solvers, matched to the plate gamma=340 extrapolation dataset (plucks,
data seed 5): same centres and widths, initial field = raised cosine, at rest.

  - FD plate:       the dataset (finite-difference plate, gamma=340, kappa=0, u = 0 at the walls)
  - modal plate:    the same PDE, boundary conditions, grid and pluck, solved exactly in the
                    sine modes of the plate (no numerical dispersion)
  - FTM modes 1-25: WaveSolver2D (the wave2d_ftm solver, rigid walls), initial pressure = pluck
  - FTM modes 0-25: the same, plus the modes with one index 0 (no static (0, 0) mode)

Writes fig_plucks.png next to this file and prints the numbers of the report.

Usage, from the repository root:
    uv run --extra cpu python docs/results/plate-vs-wave2d-spectra/plucks.py $FA2023_OUTPUT/datasets [num_ics]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.fft import rfft, rfftfreq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from platefno.solver.wave_solver_2d import WaveSolver2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FS, T, C, RHO = 48000, 2000, 340.0, 1.2041
LX, LY = 1.0, 0.95
NX, NY = 40, 38
STEPS = [0, 5, 10, 20, 39, 60, 100, 150, 200, 300, 500, 1000, 1999]
INNER = (slice(5, -5), slice(5, -5))  # away from the walls
N_FINE = 800  # grid for the projection integrals; a pluck spans only 4-8 points of the 40 x 38 grid


def raised_cosine(x, y, ctr, wid):
    """The plate's pluck (LinearPlateSolver.create_pluck) on the points x, y."""
    dist = np.sqrt((x - ctr[0]) ** 2 + (y - ctr[1] * LY) ** 2)
    return 0.5 * (dist < wid / 2) * (1 + np.cos(2 * np.pi * dist / wid))


def fine_pluck(ctr, wid):
    xf, yf = np.linspace(0, LX, N_FINE), np.linspace(0, LY, int(N_FINE * LY))
    return xf, yf, raised_cosine(xf[:, None], yf[None, :], ctr, wid)


def pluck_parameters(n):
    """Centres and widths of the plate dataset: numpy seed 5, rand(2) then rand(1) per IC."""
    np.random.seed(5)
    return [(0.6 * np.random.rand(2) + 0.2, 0.1 + np.random.rand(1)[0] * 0.1) for _ in range(n)]


class ModalPlate:
    """Exact modal solution of u_tt = c^2 lap(u), u = 0 at the walls, from rest.

    Modes sin(kx x) sin(ky y), kx = m pi / LX, ky = n pi / LY, with all the modes that the
    40 x 38 grid resolves (m <= 38, n <= 36); each mode evolves as a cos(c |k| t).
    The initial coefficients are the sine transform (Sturm-Liouville transform) of the
    continuous pluck, with the norm LX LY / 4 of each mode.
    """

    def __init__(self):
        self.kx = np.arange(1, NX - 1) * np.pi / LX
        self.ky = np.arange(1, NY - 1) * np.pi / LY
        self.omega = C * np.hypot(self.kx[:, None], self.ky[None, :])
        x, y = np.linspace(0, LX, NX), np.linspace(0, LY, NY)
        self.sx, self.sy = np.sin(np.outer(self.kx, x)), np.sin(np.outer(self.ky, y))

    def solve(self, ctr, wid):
        xf, yf, rc = fine_pluck(ctr, wid)
        sxf, syf = np.sin(np.outer(self.kx, xf)), np.sin(np.outer(self.ky, yf))
        coef = np.einsum("mx,xy,ny->mn", sxf, rc, syf) * (xf[1] - xf[0]) * (yf[1] - yf[0])
        coef = coef * 4 / (LX * LY)
        t = np.arange(T) / FS
        amp = coef[None] * np.cos(self.omega[None] * t[:, None, None])  # (T, m, n)
        return np.einsum("tmn,mx,ny->txy", amp, self.sx, self.sy)


def ftm_solver(with_index_zero):
    ws = WaveSolver2D(Fs=FS, dur=T / FS, lx=LX, ly=LY, spatial_delta=1 / NX, c0=C, modes=25)
    if not with_index_zero:
        return ws
    # The modes of WaveSolver2D.__init__ with indices 0..25, without the static (0, 0) mode
    mx, my = (a.flatten() for a in np.meshgrid(np.arange(26), np.arange(26)))
    keep = mx + my > 0
    lam_x = np.tile(mx[keep] * np.pi / LX, 2)
    lam_y = np.tile(my[keep] * np.pi / LY, 2)
    s = 1j * C * np.hypot(mx[keep] * np.pi / LX, my[keep] * np.pi / LY)
    smu = np.hstack((s, np.conj(s))) - 1  # damping 1, as in WaveSolver2D
    nmu = (-8 * lam_x**2 / (RHO * smu**2) - 8 * lam_y**2 / (RHO * smu**2) + 8 / (RHO * C**2)) * LX * LY
    # The norm of cos(0) over a side is twice the norm of cos(m pi x / L) for m > 0
    nmu = nmu * np.where((lam_x == 0) | (lam_y == 0), 2.0, 1.0)
    ws.smu, ws.nmu, ws.lamX, ws.lamY = smu, nmu, lam_x, lam_y
    ws.K1 = lambda x, y: 4 * np.cos(lam_x * x) * np.cos(lam_y * y)
    ws.K2 = lambda x, y: 4 * lam_x / (smu * RHO) * np.sin(lam_x * x) * np.cos(lam_y * y)
    ws.K3 = lambda x, y: 4 * lam_y / (smu * RHO) * np.cos(lam_x * x) * np.sin(lam_y * y)
    return ws


def ftm_pluck(ws, ctr, wid):
    """FTM pressure (T, X, Y) from an initial pressure equal to the pluck, at rest.

    The modal input is the integral of the field times Ka3; WaveSolver2D.create_impulse
    uses Ka3 at one point, which is the same integral for a delta.
    """
    xf, yf, rc = fine_pluck(ctr, wid)
    cx, cy = np.cos(np.outer(ws.lamX, xf)), np.cos(np.outer(ws.lamY, yf))
    fe = 4 * np.einsum("mx,xy,my->m", cx, rc, cy) * (xf[1] - xf[0]) * (yf[1] - yf[0])
    return np.moveaxis(ws.solve(fe)[2], -1, 0).astype(np.float64)


def representable(rc, with_index_zero):
    """The part of a field that the FTM cosine modes can represent."""
    if with_index_zero:
        return rc - rc.mean()
    return rc - rc.mean(axis=0, keepdims=True) - rc.mean(axis=1, keepdims=True) + rc.mean()


def corr(a, b, region=(slice(None), slice(None))):
    return np.corrcoef(a[region].ravel(), b[region].ravel())[0, 1]


def psd(x):
    """Power spectrum of (T, X, Y) fields, averaged over the points (Hann window)."""
    x = x - x.mean(axis=0)
    return (np.abs(rfft(x * np.hanning(len(x))[:, None, None], axis=0)) ** 2).mean(axis=(1, 2))


def quantiles(p, qs=(0.5, 0.9, 0.99)):
    c = np.cumsum(p) / p.sum()
    f = rfftfreq(T, 1 / FS)
    return " / ".join(f"{f[np.searchsorted(c, q)] / 1000:.2f}" for q in qs)


# Pairs to compare: (label, field a, field b, region of the grid)
PAIRS = [
    ("FD plate vs modal plate (numerical dispersion)", "FD plate", "modal plate", (slice(1, -1), slice(1, -1))),
    ("modal plate vs FTM modes 0-25 (boundary conditions)", "modal plate", "FTM modes 0-25", INNER),
    ("FD plate vs FTM modes 0-25 (both)", "FD plate", "FTM modes 0-25", INNER),
    ("FD plate vs FTM modes 1-25 (both, wave2d_ftm modes)", "FD plate", "FTM modes 1-25", INNER),
]


def main(datasets, num_ics):
    plate = np.load(
        f"{datasets}/gamma_340.0-kappa_0.0-damping_nodamping/ic_pluck-n_30-steps_2000-seed_5.npy",
        mmap_mode="r",
    )
    modal = ModalPlate()
    ftm = {name: ftm_solver(name == "FTM modes 0-25") for name in ("FTM modes 1-25", "FTM modes 0-25")}
    x, y = np.meshgrid(np.linspace(0, LX, NX), np.linspace(0, LY, NY), indexing="ij")
    correlations = {label: [] for label, *_ in PAIRS}
    ic_errors = {name: [] for name in ftm}
    modal_ic_error = []
    spectra = {}
    example = {}
    for i, (ctr, wid) in enumerate(pluck_parameters(num_ics)):
        rc = raised_cosine(x, y, ctr, wid)
        fields = {"FD plate": np.asarray(plate[i, :, :, :, 0], dtype=np.float64)}
        assert corr(fields["FD plate"][0], rc) > 0.99, "IC does not match the dataset"
        fields["modal plate"] = modal.solve(ctr, wid)
        inner_rc = rc[1:-1, 1:-1]
        modal_ic_error.append(np.linalg.norm(fields["modal plate"][0][1:-1, 1:-1] - inner_rc) / np.linalg.norm(inner_rc))
        for name, ws in ftm.items():
            p = ftm_pluck(ws, ctr, wid)
            target = representable(rc, name == "FTM modes 0-25")
            scale = (p[0] * target).sum() / (target**2).sum()
            ic_errors[name].append([np.linalg.norm(p[0] / scale - target) / np.linalg.norm(target),
                                    np.linalg.norm(target - rc) / np.linalg.norm(rc)])
            fields[name] = p / scale
        for label, a, b, region in PAIRS:
            correlations[label].append([corr(fields[a][k], fields[b][k], region) for k in STEPS])
        for name, f in fields.items():
            spectra[name] = spectra.get(name, 0) + psd(f)
        spectra["FD plate, v"] = spectra.get("FD plate, v", 0) + psd(np.asarray(plate[i, :, :, :, 1], dtype=np.float64))
        if i == 0:
            example = {name: f[[0, 39, 100, 200]] for name, f in fields.items()}

    print(f"Initial fields ({num_ics} ICs, median relative error):")
    print(f"  modal plate vs the pluck on the grid: {np.median(modal_ic_error):.4f}")
    for name, e in ic_errors.items():
        e = np.array(e)
        print(f"  {name}: vs the representable part {np.median(e[:, 0]):.3f}; part of the pluck outside the modes {np.median(e[:, 1]):.3f}")
    print("Correlation (median over ICs):")
    for label, c in correlations.items():
        print(f"  {label}:")
        print("    " + ", ".join(f"{k}: {v:.2f}" for k, v in zip(STEPS, np.median(c, axis=0))))
    print("Frequency (kHz) below which 50 / 90 / 99 % of the power lies:")
    for name, p in spectra.items():
        print(f"  {name}: {quantiles(p)}")
    figure(example, correlations, spectra)


def figure(example, correlations, spectra):
    names = ["FD plate", "modal plate", "FTM modes 1-25", "FTM modes 0-25"]
    steps = (0, 39, 100, 200)
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(4, 6)
    for col, name in enumerate(names):
        lim = np.abs(example[name][0]).max()
        for row, step in enumerate(steps):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(example[name][row].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(0, LX, 0, LY))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"step {step}")
    ax = fig.add_subplot(grid[0:2, 4:])
    for label, c in correlations.items():
        ax.plot(STEPS, np.median(c, axis=0), "o-", ms=4, label=label)
    ax.axvspan(0, 39, color="gray", alpha=0.15, label="training horizon")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("symlog", linthresh=10)
    ax.set(xlim=(0, 2100), xlabel="time step", ylabel="correlation (median over ICs)", ylim=(-0.9, 1.02),
           title="Correlation of the fields (FTM pairs: away from the walls)")
    ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(grid[2:, 4:])
    f = rfftfreq(T, 1 / FS) / 1000
    for name, p in spectra.items():
        ax.plot(f, np.cumsum(p) / p.sum(), ls=":" if name == "FD plate, v" else "-",
                label=name + (" (velocity)" if name == "FD plate, v" else ""))
    ax.set(xlim=(0, 7), xlabel="frequency (kHz)", ylabel="cumulative power",
           title="Temporal spectra of the plucks (displacement / pressure)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_plucks.png", dpi=100)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
