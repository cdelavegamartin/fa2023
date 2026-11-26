import numpy as np
from scipy.fft import dstn, idstn

from platefno.solver.linear_plate_solver import LinearPlateSolver


class ModalPlateSolver(LinearPlateSolver):
    """Exact modal solution of the plate of LinearPlateSolver, without numerical dispersion.

    The PDE is the same: u_tt = gamma^2 lap(u) - kappa^2 lap(lap(u)) - 2 sig0 u_t
    + 2 sig1 lap(u_t), with u = 0 and lap(u) = 0 at the edges (simply supported), on the
    same grid, with the same damping parameters and initial conditions.
    The finite-difference solver replaces lap by a 5-point stencil, which makes the
    frequencies of the short waves too low. This solver expands the initial state in the
    sine modes sin(kx x) sin(ky y) that the grid resolves (a DST-I of the interior points,
    which reproduces the initial state on the grid exactly) and evolves each mode with its
    exact frequency and decay:

        Lambda = kx^2 + ky^2,  sigma = sig0 + sig1 Lambda,
        beta = gamma^2 Lambda + kappa^2 Lambda^2,  omega = sqrt(beta - sigma^2).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kx = np.arange(1, self.Nx - 1) * np.pi  # the plate is [0, 1] x [0, aspect_ratio]
        ky = np.arange(1, self.Ny - 1) * np.pi / self.aspect_ratio
        lam = kx[:, None] ** 2 + ky[None, :] ** 2
        self.sigma = self.sig0 + self.sig1 * lam
        self.beta = self.gamma**2 * lam + self.kappa**2 * lam**2
        # complex for overdamped modes (beta < sigma^2); the solution stays real
        self.omega = np.sqrt(self.beta - self.sigma**2 + 0j)

    def solve(self, wb0):
        """Same input and output as LinearPlateSolver.solve: (u, v, t), u and v (Nx, Ny, numT)."""
        u0 = wb0[: self.ss].reshape(self.Nx, self.Ny)[1:-1, 1:-1]
        v0 = wb0[self.ss :].reshape(self.Nx, self.Ny)[1:-1, 1:-1]
        a = dstn(u0, type=1, norm="ortho")
        b = dstn(v0, type=1, norm="ortho")

        t = self.t_eval[:, None, None]
        decay = np.exp(-self.sigma * t)
        cos = np.cos(self.omega * t)
        # sin(omega t) / omega, with its limit t for omega = 0
        omega = np.where(self.omega == 0, 1, self.omega)
        sinc = np.where(self.omega == 0, t, np.sin(omega * t) / omega)
        u_modal = np.real(decay * (a * cos + (b + self.sigma * a) * sinc))
        v_modal = np.real(decay * (b * cos - (self.sigma * b + self.beta * a) * sinc))

        u = np.zeros((self.Nx, self.Ny, self.numT))
        v = np.zeros((self.Nx, self.Ny, self.numT))
        u[1:-1, 1:-1] = np.moveaxis(idstn(u_modal, type=1, axes=(1, 2), norm="ortho"), 0, -1)
        v[1:-1, 1:-1] = np.moveaxis(idstn(v_modal, type=1, axes=(1, 2), norm="ortho"), 0, -1)
        return u.astype(np.float32), v.astype(np.float32), self.t_eval.astype(np.float32)
