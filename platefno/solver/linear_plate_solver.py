import numpy as np
from scipy.sparse import diags, kron, eye, bsr_array
from scipy.integrate import solve_ivp


class LinearPlateSolver:
    def __init__(
        self,
        SR=96000,
        TF=1.0,
        gamma=1.0,
        kappa=1.0,
        t60=({"f": 100, "T60": 5}, {"f": 2000, "T60": 3}),
        aspect_ratio=1.0,
        Nx=100,  # number of grid points in x direction (including boundary points)
        bc="clamped",
    ):
        super(LinearPlateSolver, self).__init__()

        self.SR = SR
        self.TF = TF
        self.gamma = gamma
        self.kappa = kappa
        self.aspect_ratio = aspect_ratio
        self.Nx = Nx
        self.bc = bc

        # If t60 is a tuple and has two elements, then it is a frequency dependent
        # t60. If it is a single number, then it is a constant t60.
        if isinstance(t60, tuple) and len(t60) == 2:
            # check that the two elements are dicts
            if isinstance(t60[0], dict) and isinstance(t60[1], dict):
                self.sig0, self.sig1 = self._t60_freq_dep(t60)
            else:
                raise ValueError("t60 must be a tuple of dicts or a number")

        elif isinstance(t60, (int, float)):
            self.sig0 = 6 * np.log(10) / (t60)
            self.sig1 = 0
        elif t60 == None:
            self.sig0 = 0
            self.sig1 = 0
        else:
            raise ValueError("t60 must be a tuple of dicts or a number")

        # derived parameters

        k = 1 / self.SR  # time step
        self.numT = int(np.floor(self.SR * self.TF))  # duration of simulation (samples)
        self.Ny = int(np.floor((self.Nx - 1) * self.aspect_ratio) + 1)
        hx = 1 / (self.Nx - 1)
        hy = self.aspect_ratio / (self.Ny - 1)
        self.hx = hx
        self.hy = hy

        self.ss = (self.Nx) * (self.Ny)  # total grid size flattened

        # meshgrid
        self.X, self.Y = np.meshgrid(np.arange(self.Nx) * hx, np.arange(self.Ny) * hy)
        self.t_eval = np.arange(0, self.numT) * k

        # create finite difference matrices
        # Dxx
        Dxx = diags(
            [1, -2, 1], [-1, 0, 1], shape=(self.Nx, self.Nx)
        ).toarray()  # Create sparse matrix Dxx
        Dxx /= hx**2  # Divide matrix Dxx by h**2
        # Dyy
        Dyy = diags(
            [1, -2, 1], [-1, 0, 1], shape=(self.Ny, self.Ny)
        ).toarray()  # Create sparse matrix Dyy
        Dyy /= hy**2  # Divide matrix Dyy by h**2

        # Implement boundary conditions
        # The way these are implemented results in the same Dxx, Dyy matrices
        # for both clamped and simply supported boundary conditions.
        #  I think this is correct considering the difference matrix is second order,
        # as well as the the discretisation conditions applied
        # It would therefore more efficient to not ask, and also use a grid of
        # N-1 points on each side.
        # I don't know if that makes a performance difference using sparse matrices
        if self.bc == "clamped":
            # # Implement Neumann boundary conditions (centered)
            Dxx[0, 1] = 2
            Dxx[-1, -2] = 2
            Dyy[0, 1] = 2
            Dyy[-1, -2] = 2

            # Implement dirichlet boundary conditions
            Dxx[0, :] = 0
            Dxx[-1, :] = 0
            Dyy[0, :] = 0
            Dyy[-1, :] = 0

        elif self.bc == "simply-supported":
            # Implement dxxu0 = 0 boundary conditions (second order, centered)
            Dxx[0, 0] = 0
            Dxx[-1, 0] = 0
            Dyy[0, 0] = 0
            Dyy[-1, 0] = 0

            # Implement dirichlet boundary conditions
            Dxx[0, :] = 0
            Dxx[-1, :] = 0
            Dyy[0, :] = 0
            Dyy[-1, :] = 0

        Dxx2 = kron(Dxx, eye(self.Ny))
        Dyy2 = kron(eye(self.Nx), Dyy)
        D = Dxx2 + Dyy2

        DD = D @ D
        self.A = bsr_array(-self.kappa**2 * DD + self.gamma**2 * D)
        self.B = bsr_array(-2 * self.sig0 * eye(self.ss) + 2 * self.sig1 * D)

    def _t60_freq_dep(self, t60):
        """Calculate the frequency dependent sig0 and sig1 parameters."""
        # Extract the frequencies and t60 values
        f = np.array([t60[0]["f"], t60[1]["f"]])
        T60 = np.array([t60[0]["T60"], t60[1]["T60"]])
        # check that frequencies are different and ordered from smallest to largest
        if f[0] > f[1]:
            #  reverse order
            f = f[::-1]
            T60 = T60[::-1]
        elif f[0] == f[1]:
            raise ValueError("frequencies must be different")

        if T60[0] < T60[1]:
            raise ValueError("T60 must be decreasing with frequency")

        # Calculate the sig0 and sig1 parameters
        sig0 = (
            6
            * np.log(10)
            / (self._eta_func(f[1]) - self._eta_func(f[0]))
            * (self._eta_func(f[1]) / T60[0] - self._eta_func(f[0]) / T60[1])
        )

        sig1 = (
            6
            * np.log(10)
            / (self._eta_func(f[1]) - self._eta_func(f[0]))
            * (-1 / T60[0] + 1 / T60[1])
        )

        return sig0, sig1

    def _eta_func(self, frequency):
        """Calculate the eta function for a given frequency."""
        omega = 2 * np.pi * frequency
        # calculate eta, depending if kappa is zero or not
        if self.kappa == 0:
            eta = omega**2 / self.gamma**2
        else:
            eta = (
                -self.gamma**2
                + np.sqrt(self.gamma**4 + 4 * self.kappa**2 * omega**2)
            ) / (2 * self.kappa**2)
        return eta

    def create_pluck(self, ctr, wid, u0_max=1.0, v0_max=0.0):
        # create raised cosine
        # calculate disnces from center

        dist = np.sqrt(
            (self.X.transpose() - ctr[0]) ** 2
            + (self.Y.transpose() - ctr[1] * self.aspect_ratio) ** 2
        )

        ind = np.sign(np.maximum((-dist + wid / 2.0), 0))
        rc = 0.5 * ind * (1 + np.cos(2 * np.pi * dist / wid))

        # make sure the boundary conditions are met in the initial conditions
        if self.bc == "clamped" or self.bc == "simply-supported":
            rc[0, :] = 0
            rc[-1, :] = 0
            rc[:, 0] = 0
            rc[:, -1] = 0

        self.u0 = u0_max * rc
        self.v0 = v0_max * rc

        w0 = np.concatenate(
            (
                self.u0.reshape(-1, 1),
                self.v0.reshape(-1, 1),
            )
        )
        w0 = np.squeeze(w0, axis=-1)
        return w0

    def create_random_initial(self, u0_max=1.0, v0_max=0.0):
        # create raised cosine
        # calculate disnces from center

        rand_init = 2.0 * np.random.rand(self.Nx, self.Ny) - 1.0

        # make sure the boundary conditions are met in the initial conditions
        if self.bc == "clamped" or self.bc == "simply-supported":
            rand_init[0, :] = 0
            rand_init[-1, :] = 0
            rand_init[:, 0] = 0
            rand_init[:, -1] = 0

        self.u0 = u0_max * rand_init
        self.v0 = v0_max * rand_init

        w0 = np.concatenate(
            (
                self.u0.reshape(-1, 1),
                self.v0.reshape(-1, 1),
            )
        )
        w0 = np.squeeze(w0, axis=-1)
        return w0

    def linplate(self, t1, wb):
        # extract parameter values from self
        ss = self.ss
        A = self.A

        # extract variables from wb
        u = wb[:ss]
        v = wb[ss : 2 * ss]

        du_dt = v
        dv_dt = A @ u + self.B @ v

        dw_dt = np.concatenate((du_dt, dv_dt))
        return dw_dt

    def solve(self, wb0):
        tspan = [0, self.TF]  # time span for ode
        sol = solve_ivp(
            self.linplate,
            tspan,
            wb0,
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-12,
        )

        t1 = sol.t
        wb1 = sol.y
        self.wb1 = wb1
        self.t1 = t1
        ss = self.ss
        u = wb1[:ss, :]
        v = wb1[ss : 2 * ss, :]

        u = u.reshape(
            self.Nx,
            self.Ny,
            self.numT,
        )
        v = v.reshape(
            self.Nx,
            self.Ny,
            self.numT,
        )

        # Convert to float32
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        t1 = t1.astype(np.float32)

        return u, v, t1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # define parameters for initial conditions
    ctr = (0.3, 0.7)
    wid = 0.2
    u0_max = 10
    v0_max = 0

    # t60 = ({"f": 100, "T60": 5}, {"f": 2000, "T60": 3})
    t60 = None

    # time the solver
    start = time.time()

    # create solver
    solver = LinearPlateSolver(
        SR=48000,
        TF=0.01,
        gamma=1.0,
        kappa=1.0,
        t60=t60,
        aspect_ratio=1.1,
        Nx=80,
    )
    # print sig0 and sig1
    print(solver.sig0, solver.sig1)

    # create initial conditions
    wb0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
    # wb0 = solver.create_random_initial(u0_max, v0_max)
    # solve
    u, v, t1 = solver.solve(wb0)

    # print time
    print("time = ", time.time() - start)

    # Assert that the t1 array is the same as the t_eval array
    assert np.allclose(t1, solver.t_eval)
    # Print first and last time values, labelled in a f-string
    print(f"t1[0] = {t1[0]:.3f}, t1[-1] = {t1[-1]:.3f}")

    # Print shape of X, Y, u, v
    print(u.shape)
    print(v.shape)
    print(solver.X.shape)
    print(solver.Y.shape)
    # # Assert that the u and v arrays are the same shape as the X and Y arrays
    # assert u.shape == solver.X.shape
    # assert u.shape == solver.Y.shape
    # assert v.shape == solver.X.shape
    # assert v.shape == solver.Y.shape

    # Print grid spacing hx, hy and total side lengths Lx, Ly
    print(f"hx = {solver.hx:.3f}, hy = {solver.hy:.3f}")
    print(f"Lx = {(solver.Nx-1)*solver.hx:.3f}, Ly = {(solver.Ny-1)*solver.hy:.3f}")

    assert np.allclose(u[..., 0], solver.u0)
    assert np.allclose(v[..., 0], solver.v0)
    # Asserts to check the unravelling and rearranging of the wb1 array
    xi = np.linspace(0, solver.Nx - 1, solver.Nx, dtype=int)
    yi = np.linspace(0, solver.Ny - 1, solver.Ny, dtype=int)

    def assert_correct_rearranging(xi, yi, solver):
        for i in range(len(xi)):
            for j in range(len(yi)):
                wi = solver.Ny * xi[i] + yi[j]
                # print("Coords: ", f"x={xi[i]}", f"y={yi[j]}", f"w={wi}")
                assert np.allclose(u[xi[i], yi[j], :], solver.wb1[wi, :])
                assert np.allclose(v[xi[i], yi[j], :], solver.wb1[solver.ss + wi, :])
        return

    assert_correct_rearranging(xi, yi, solver)

    # Print the shape of X, Y from the solver
    print(solver.X.shape)
    print(solver.Y.shape)

    # get maximum speed
    v_max = np.max(np.abs(v))
    print("v_max = ", v_max)
    # get maximum displacement
    u_max = np.max(np.abs(u))
    print("u_max = ", u_max)

    def countnonzero(arr):
        return arr.size - np.count_nonzero(arr)

    # Print number of nonzero elements in u0 and u[..., 0]
    print("u0 nonzero = ", countnonzero(solver.u0))
    print("u[..., 0] nonzero = ", countnonzero(u[..., 0]))

    # fig_width = 237 / 72.27  # Latex columnwidth expressed in inches
    fig_width = 20
    figsize = (fig_width, fig_width * 0.75)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.0, wspace=0.05)
    axs = gs.subplots(sharex="row", sharey=True)

    print(u.shape)
    axs[0, 0].imshow(
        solver.u0.transpose(),
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        cmap="viridis",
    )

    # for (j, i), label in np.ndenumerate(solver.u0.transpose()):
    #     axs[0, 0].text(i, j, f"{label:.3f}", ha="center", va="center")

    axs[0, 1].imshow(
        u[..., 0].transpose(),
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        cmap="viridis",
    )

    # for (j, i), label in np.ndenumerate(u[..., 0].transpose()):
    #     axs[0, 1].text(i, j, f"{label:.3f}", ha="center", va="center")

    axs[0, 2].imshow(
        u[..., -1].transpose(),
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        cmap="viridis",
    )

    # for (j, i), label in np.ndenumerate(u[..., -1].transpose()):
    #     axs[0, 2].text(i, j, f"{label:.3f}", ha="center", va="center")

    axs[1, 0].imshow(
        u[..., u.shape[-1] // 2].transpose(),
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        cmap="viridis",
    )
    plt.show()
