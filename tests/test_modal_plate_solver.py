import numpy as np
import pytest

from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.solver.modal_plate_solver import ModalPlateSolver

FS = 48000
T60 = ({"f": 100, "T60": 5}, {"f": 2000, "T60": 3})


def solvers(steps, **kwargs):
    args = dict(SR=FS, TF=steps / FS, aspect_ratio=0.95, Nx=40, **kwargs)
    return LinearPlateSolver(**args), ModalPlateSolver(**args)


def eigenmode(solver, m, n):
    """Initial state (u = sin(m pi x) sin(n pi y / ly), at rest) of the solver."""
    u0 = np.sin(m * np.pi * solver.X.T) * np.sin(n * np.pi * solver.Y.T / solver.aspect_ratio)
    return np.concatenate((u0.ravel(), np.zeros(solver.ss)))


def test_reproduces_the_initial_state():
    _, modal = solvers(3, gamma=340.0, kappa=0.0, t60=None)
    w0 = modal.create_pluck(np.array([0.4, 0.6]), 0.15)
    u, v, _ = modal.solve(w0)
    assert np.allclose(u[..., 0], w0[: modal.ss].reshape(modal.Nx, modal.Ny), atol=1e-6)
    assert np.allclose(v[..., 0], 0, atol=1e-6)


@pytest.mark.parametrize("gamma, kappa, t60", [(340.0, 0.0, None), (100.0, 1.0, T60), (1.0, 0.1, T60)])
def test_agrees_with_finite_differences_on_a_low_mode(gamma, kappa, t60):
    # the 5-point stencil is accurate for the (1, 1) mode, so both solvers agree
    fd, modal = solvers(40, gamma=gamma, kappa=kappa, t60=t60)
    w0 = eigenmode(fd, 1, 1)
    u_fd, v_fd, _ = fd.solve(w0)
    u_m, v_m, _ = modal.solve(w0)
    assert np.abs(u_m - u_fd).max() < 1e-3 * np.abs(u_fd).max()
    assert np.abs(v_m - v_fd).max() < 1e-3 * np.abs(v_fd).max()


def test_high_mode_has_the_exact_frequency():
    # the (30, 30) mode of the undamped wave equation: exact omega = c |k|; the
    # finite-difference solver is slower, the modal solver is not
    steps = 200
    fd, modal = solvers(steps, gamma=340.0, kappa=0.0, t60=None)
    w0 = eigenmode(modal, 30, 30)
    omega = 340.0 * np.pi * np.hypot(30, 30 / 0.95)
    t = np.arange(steps) / FS
    i, j = 20, 19
    u_m, _, _ = modal.solve(w0)
    expected = w0[: modal.ss].reshape(modal.Nx, modal.Ny)[i, j] * np.cos(omega * t)
    assert np.abs(u_m[i, j] - expected).max() < 1e-4
    u_fd, _, _ = fd.solve(w0)
    assert np.abs(u_fd[i, j] - expected).max() > 0.1


def test_undamped_energy_is_constant():
    _, modal = solvers(500, gamma=340.0, kappa=0.0, t60=None)
    u, v, _ = modal.solve(modal.create_pluck(np.array([0.3, 0.5]), 0.12))
    # modal energy: sum over modes of (v_m^2 + beta u_m^2) / 2, from a DST-I of the grid
    from scipy.fft import dstn

    um = dstn(np.moveaxis(u[1:-1, 1:-1], -1, 0).astype(np.float64), type=1, axes=(1, 2), norm="ortho")
    vm = dstn(np.moveaxis(v[1:-1, 1:-1], -1, 0).astype(np.float64), type=1, axes=(1, 2), norm="ortho")
    energy = 0.5 * (vm**2 + modal.beta * um**2).sum(axis=(1, 2))
    assert np.abs(energy / energy[0] - 1).max() < 1e-4
