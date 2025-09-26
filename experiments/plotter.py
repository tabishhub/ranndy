from enum import Enum, EnumMeta

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import cm, colors


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class AvailableOperators(str, BaseEnum):
    KOOPMAN = "koopman"
    KOOPMAN_GENERATOR = "koopman_generator"
    SCHRODINGER = "schrodinger"
    FORWARD_BACKWARD = "forward_backward"


# plt.rcParams.update(get_rcparams())


def get_rcparams():
    SMALL_SIZE = 8
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Times"], "size": SMALL_SIZE}

    rc_params = {
        "axes.titlesize": SMALL_SIZE,  # fontsize of the axes title
        "axes.labelsize": SMALL_SIZE,  # fontsize of the x and y labels
        "xtick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
        "ytick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
        "legend.fontsize": SMALL_SIZE,  # legend fontsize
        "figure.titlesize": BIGGER_SIZE,  # fontsize of the figure title
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,  # Disable grid
        "axes.spines.top": False,  # Hide top spine
        "axes.spines.right": False,  # Hide right spine
        "axes.spines.left": True,  # Show left spine
        "axes.spines.bottom": True,  # Show bottom spine
        "xtick.bottom": True,  # Ensure x-ticks are shown
        "ytick.left": True,  # Ensure y-ticks are shown
    }

    return rc_params


# TODO: modify this function and use this
def set_size(
    width: float | str = 360.0,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
    adjust_height_for_label: float | None = None,
) -> tuple:
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if adjust_height_for_label is None:
        golden_ratio = (5**0.5 - 1) / 2

    else:
        golden_ratio = ((5**0.5 - 1) / 2) + adjust_height_for_label

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


class PlottingResults:
    def __init__(self, X, bounds, boxes, n_eigfuncs, operator):
        self.X = X
        self.bounds = bounds
        self.boxes = boxes
        self.n_eigfuncs = n_eigfuncs
        self.operator = operator
        self.h = np.divide(bounds[:, 1] - bounds[:, 0], boxes)
        self.dimension = boxes.size
        self.numVertices = (self.boxes + 1).prod()
        self.numBoxes = self.boxes.prod

        # plt.rcParams(get_rcparams())

    # def dimension(self):
    #     '''
    #     Returns dimension of the domain.
    #     '''
    #     return self.eigvals

    def midpointGrid(self):
        """
        Returns a grid given by the midpoints of the boxes.
        """
        b = self.bounds
        h = self.h
        n = self.numBoxes()
        x = []
        for i in range(self.dimension):
            x.append(np.linspace(b[i, 0] + h[i] / 2, b[i, 1] - h[i] / 2, self.boxes[i]))
        X = np.meshgrid(*x, indexing="ij")
        c = np.zeros([self.dimension, n])
        for i in range(self.dimension):
            c[i, :] = X[i].reshape(n)
        return c

    def vertexGrid(self):
        """
        Returns a grid given by the vertices.
        """
        b = self.bounds
        n = self.numVertices()
        x = []
        for i in range(self.dimension):
            x.append(np.linspace(b[i, 0], b[i, 1], self.boxes[i] + 1))
        X = np.meshgrid(*x, indexing="ij")
        c = np.zeros([self.dimension, n])
        for i in range(self.dimension):
            c[i, :] = X[i].reshape(n)
        isBoundary = np.zeros(X[i].shape, dtype=bool)
        for i in range(self.dimension):
            ind = self.dimension * [slice(None)]
            ind[i] = [0, self.boxes[i]]
            isBoundary[tuple(ind)] = True
        isBoundary = isBoundary.reshape(n)
        return c, isBoundary

    def get_c(self, grid="midpoint"):
        if self.dimension > 3:
            print("Not defined for d > 3.")

        if grid == "midpoint":
            c = self.midpointGrid()
            dims = self.boxes
        elif grid == "vertex":
            c, _ = self.vertexGrid()
            dims = self.boxes + 1
        elif grid == "data":
            c = self.X
            dims = self.X  # TODO: check this
        else:
            print("Grid not defined")

        return c, dims

    def plot(self, v, ax, mode="2D", grid="midpoint", **plot_kwargs):
        """
        Plots v, where v_i is the value at grid point i.

        :param mode: select plot type for two-dimensional domains.
        :param grid: select grid type (midpoint or vertex)
        """
        c, dims = self.get_c(grid=grid)

        getattr(self, "plot_%s" % self.dimension)(c, v, dims, mode, ax, **plot_kwargs)

    def plot_1(self, c, v, dims, mode, ax, **plot_kwargs):
        c = c.squeeze()
        ax.plot(c, v, **plot_kwargs)

    def plot_2(self, c, v, dims, mode, ax, **plot_kwargs):
        X = c[0, :].reshape(dims)
        Y = c[1, :].reshape(dims)
        Z = v.reshape(dims)

        if mode == "2D":
            ax.pcolor(X, Y, Z, **plot_kwargs)
        else:
            # ax = plt.gcf().add_subplot(projection="3d")
            ax.plot_surface(X, Y, Z, **plot_kwargs)
            # ax.set_xlabel("x_1")
            # ax.set_ylabel("x_2")

    def plot_3(self, c, v, dims, mode, ax, **plot_kwargs):
        X = c[0, :].reshape(dims)
        Y = c[1, :].reshape(dims)
        Z = c[2, :].reshape(dims)
        V = v.reshape(dims)

        # ax = plt.gcf().add_subplot(projection= "3d")
        ax.scatter(X, Y, Z, c=V, **plot_kwargs)
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_zlabel("x_3")

    def plot_potential(self, system, hist=False):
        a = np.arange(int(self.bounds[0][0]), int(self.bounds[0][1]), 0.01)
        b = np.arange(int(self.bounds[1][0]), int(self.bounds[1][1]), 0.01)

        # V = generate_surface(a, b)

        if system == "lemon-slice":
            xy = np.meshgrid(a, b)
            V = generate_surface(xy[0], xy[1])
        else:
            xy = np.meshgrid(a, b)
            V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        if hist:
            if self.dimension == 1:
                plt.hist(self.X.T, bins=100)
            elif self.dimension == 2:
                plt.hist2d(self.X[0, :], self.X[1, :], bins=100)

        else:
            con = ax.contourf(xy[0], xy[1], V, levels=40, cmap="coolwarm")

        return fig, ax, con

    def plot_eigenvalues(
        self,
        eigvals,
        **plot_kwargs,
    ):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        xv = jnp.arange(1, self.n_eigfuncs + 1, 1)
        ax.plot(xv, eigvals, "o")

        return fig, ax

    def plot_eigenvalues_uncertainty(
        self,
        eigvals,
        num_eigvals_to_plot=3,
        eigvals_plus=None,
        eigvals_minus=None,
        **plot_kwargs,
    ):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        xv = jnp.arange(1, num_eigvals_to_plot + 1, 1)
        ax.plot(xv, eigvals, "o", label="Eigenvalues")
        uncertainty = np.array(eigvals_plus) - np.array(eigvals)
        norm = colors.Normalize(vmin=min(uncertainty), vmax=max(uncertainty))
        cmap = cm.get_cmap("coolwarm")
        for i in range(len(xv)):
            color = cmap(norm(uncertainty[i]))
            ax.plot(
                [xv[i], xv[i]],
                [eigvals_minus[i], eigvals_plus[i]],
                color=color,
                linewidth=6,
                solid_capstyle="round",
            )

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)

        return fig, ax, cbar

    def plot_eigenfunctions(
        self,
        eigfuncs,
        num_funcs_to_plot=3,
        mode="3D",
        grid="midpoint",
        comparsion=False,
        **kwargs,
    ):
        if self.dimension == 1:
            # markers = ["o", "s", "^"]
            lines = ["-", "--", ":"]
            # cmap = cm.get_cmap("tab10", num_funcs_to_plot)  # or "tab20", "Set1", etc.
            # colors = [cmap(i) for i in range(num_funcs_to_plot)]
            colors = ["red", "green", "blue"]
            # labels = ["RaNNDy", "VAMPnets", "Exact"]
            fig, ax = plt.subplots()  # Explicitly create Axes
            for i in range(num_funcs_to_plot):
                self.plot(
                    np.real(eigfuncs[i, :] / np.amax(np.abs(eigfuncs[i, :]))),
                    ax=ax,
                    linestyle=lines[0],
                    # linewidth=0.1,
                    alpha=0.7,
                    # marker=markers[0],
                    markersize=0.8,
                    color=colors[i],
                    label=rf"$\varphi_{i + 1}$",  # labels[0] if i == 0 else None,
                )
            if comparsion:
                eigfuncs2 = kwargs.get("eigf_vampnets")
                eigfuncs3 = kwargs.get("eigf_exact")
                for i in range(num_funcs_to_plot):
                    self.plot(
                        np.real(eigfuncs2[i, :] / np.amax(np.abs(eigfuncs2[i, :]))),
                        ax=ax,
                        # linewidth=0.1,
                        linestyle=lines[1],
                        alpha=0.7,
                        # marker=markers[1],
                        markersize=0.8,
                        color=colors[i],
                        # label=labels[1] if i == 0 else None,
                    )
                for i in range(num_funcs_to_plot):
                    self.plot(
                        np.real(eigfuncs3[i, :] / np.amax(np.abs(eigfuncs3[i, :]))),
                        ax=ax,
                        # linewidth=0.1,
                        linestyle=lines[2],
                        alpha=0.7,
                        # marker=markers[2],
                        markersize=0.8,
                        color=colors[i],
                        # label=labels[2] if i == 0 else None,
                    )

        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            self.plot(
                np.real(
                    eigfuncs[num_funcs_to_plot, :]
                    / np.amax(abs(eigfuncs[num_funcs_to_plot, :]))
                ),
                ax,
                mode=mode,
                grid=grid,
                cmap=cm.coolwarm,
                alpha=0.9,
                linewidth=0,
            )

        return fig, ax

    def plot_eigenfunctions_uncertainty(
        self,
        avg_eigfuncs,
        eigfuncs_plus,
        eigfuncs_minus,
        num_funcs_to_plot=3,
        trend=False,
        mode="3D",
        grid="midpoint",
    ):
        if self.dimension == 1:
            fig, ax = plt.subplots()  # Explicitly create Axes
            for i in range(num_funcs_to_plot):
                avg = np.real(avg_eigfuncs[i, :])
                plus = np.real(eigfuncs_plus[i, :])
                minus = np.real(eigfuncs_minus[i, :])

                norm = np.amax(np.abs(avg))
                avg /= norm

                local_uncertainty = np.abs(plus - minus) / 2
                local_uncertainty /= np.max(local_uncertainty)

                # scatter = ax.scatter(domain_eigf, local_uncertainty, label=f'{i}')

                self.plot(
                    np.real(avg_eigfuncs[i, :] / np.amax(np.abs(avg_eigfuncs[i, :]))),
                    ax=ax,
                    s=0.2,
                    alpha=0.7,
                )
                ax.fill_between(
                    self.c.squeeze(),
                    avg - local_uncertainty,
                    avg + local_uncertainty,
                    color="green",
                    alpha=0.3,
                    label="Uncertainty",
                )
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            avg = np.real(avg_eigfuncs[num_funcs_to_plot, :])
            plus = np.real(eigfuncs_plus[num_funcs_to_plot, :])
            minus = np.real(eigfuncs_minus[num_funcs_to_plot, :])

            norm = np.amax(np.abs(avg))
            avg /= norm

            local_uncertainty = np.abs(plus - minus) / 2
            local_uncertainty /= np.max(local_uncertainty)

            if trend:
                if mode == "2D":
                    ax, fig = fig, ax = plt.subplots()

                self.plot(
                    local_uncertainty,
                    ax,
                    mode=mode,
                    grid=grid,
                    # cmap=cm.coolwarm,
                    alpha=0.9,
                    linewidth=0,
                )

            else:
                self.plot(
                    avg,
                    ax,
                    mode=mode,
                    grid=grid,
                    cmap=cm.coolwarm,
                    alpha=0.9,
                    linewidth=0,
                )
                self.plot(
                    avg - local_uncertainty,
                    ax,
                    mode=mode,
                    grid=grid,
                    color="grey",
                    alpha=0.4,
                    linewidth=0,
                )
                self.plot(
                    avg + local_uncertainty,
                    ax,
                    mode=mode,
                    grid=grid,
                    color="grey",
                    alpha=0.4,
                    linewidth=0,
                )

        return fig, ax

    def cluster_eigenfunctions(
        self,
        eigfuncs,
        n_eigfuncs: int,
        n_clusters: int,
        grid="midpoint",
        iters: int = 1000,
    ):
        c, _ = self.get_c(grid=grid)
        cluster_colors = ["blue", "green", "cyan", "orange", "purple"]

        cc, ll = sp.cluster.vq.kmeans2(
            np.real(eigfuncs[:n_eigfuncs, :]).T, n_clusters, iter=iters
        )
        cols = [cluster_colors[label] for label in ll]
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.scatter(c[0, :], c[1, :], c=cols, alpha=0.7, edgecolor="k")

        return fig, ax


# utilities functions

## Define the exact eigenvalues and eigenfunctions for the OU process and few other functions

exact_eigv = [np.exp(-0.5 * i) for i in range(5)]


def exact_eigf_ou(x, i):
    if i == 0:  # Exact eigenfunctions
        return np.ones(x.shape[1])
    if i == 1:
        return 2 * x
    if i == 2:
        return (4 * x**2 - 1) / np.sqrt(2)
    if i == 3:
        return (8 * x**3 - 6 * x) / np.sqrt(6)
    if i == 4:
        return (16 * x**4 - 24 * x**2 + 3) / np.sqrt(24)


## lemon slice potential
def lemon_slice_potential(x, y, n=5):
    return np.cos(n * np.arctan2(y, x)) + 10 * (np.sqrt(x**2 + y**2) - 1) ** 2


def generate_surface(x, y):
    potential = np.vectorize(lemon_slice_potential)
    return potential(x, y)


def make_eigf_same_sign(domain_eigf, eigf1, eigf2):
    # Exact eigenfunctions
    true_eigf_ou = np.zeros((5, domain_eigf.shape[1]))
    for i in range(5):
        true_eigf_ou[i, :] = exact_eigf_ou(domain_eigf, i)
    # Making the sign of eigenfunctions same as in exact eigenfunctions

    for i in range(eigf1.shape[0]):
        V_exact = true_eigf_ou[i, :]
        V_1 = eigf1[i, :]
        V_2 = -eigf1[i, :]
        e1 = np.linalg.norm(V_exact - V_1)
        e2 = np.linalg.norm(V_exact - V_2)
        if e1 > e2:
            eigf1 = eigf1.at[i, :].set(-eigf1[i, :])

    # Making the sign of eigenfunctions same as in exact eigenfunctions

    for i in range(eigf2.shape[0]):
        V_exact = true_eigf_ou[i, :]
        V_1 = eigf2[i, :]
        V_2 = -eigf2[i, :]
        e1 = np.linalg.norm(V_exact - V_1)
        e2 = np.linalg.norm(V_exact - V_2)
        if e1 > e2:
            eigf2 = eigf2.at[i, :].set(-eigf2[i, :])

    return true_eigf_ou, eigf1, eigf2
