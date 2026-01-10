"""Plotting utilities for special functions (Bessel, Legendre, etc.)."""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, legendre, hermite
from numpy.typing import NDArray

from diffeq.core.plotting import PlotManager
from diffeq.core.config import PlotConfig, DEFAULT_PLOT_CONFIG


class SpecialFunctionPlotter:
    """Utilities for plotting special functions."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize special function plotter.

        Args:
            config: Plot configuration. If None, uses default.
        """
        self.plot_manager = PlotManager(config=config or DEFAULT_PLOT_CONFIG)

    def plot_bessel(
        self,
        orders: list[int],
        x: NDArray[np.floating],
        kind: str = "j",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot Bessel functions of the first or second kind.

        Args:
            orders: List of orders (n values) to plot.
            x: x-coordinates.
            kind: 'j' for J_n (first kind) or 'y' for Y_n (second kind).

        Returns:
            Tuple of (figure, axes).
        """
        fig, ax = self.plot_manager.create_figure()

        for i, n in enumerate(orders):
            if kind == "j":
                y = jv(n, x)
            elif kind == "y":
                y = yv(n, x)
            else:
                raise ValueError(f"Invalid kind: {kind}")

            color = self.plot_manager.config.get_color(i)
            ax.plot(x, y, label=f"${kind.upper()}_{{{n}}}(x)$", color=color)

        ax.set_xlabel("$x$")
        ax.set_ylabel(f"${kind.upper()}_n(x)$")
        ax.set_title(f"Bessel Functions of the {('First' if kind == 'j' else 'Second')} Kind")
        ax.legend()
        ax.grid(True, alpha=self.plot_manager.config.grid_alpha)

        return fig, ax

    def plot_legendre(
        self,
        orders: list[int],
        x: NDArray[np.floating],
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot Legendre polynomials.

        Args:
            orders: List of orders (n values) to plot.
            x: x-coordinates (should be in [-1, 1]).

        Returns:
            Tuple of (figure, axes).
        """
        fig, ax = self.plot_manager.create_figure()

        for i, n in enumerate(orders):
            P = legendre(n)
            y = P(x)
            color = self.plot_manager.config.get_color(i)
            ax.plot(x, y, label=f"$P_{{{n}}}(x)$", color=color)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$P_n(x)$")
        ax.set_title("Legendre Polynomials")
        ax.legend()
        ax.grid(True, alpha=self.plot_manager.config.grid_alpha)
        ax.set_xlim(-1.1, 1.1)

        return fig, ax

    def plot_hermite(
        self,
        orders: list[int],
        x: NDArray[np.floating],
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot Hermite polynomials.

        Args:
            orders: List of orders (n values) to plot.
            x: x-coordinates.

        Returns:
            Tuple of (figure, axes).
        """
        fig, ax = self.plot_manager.create_figure()

        for i, n in enumerate(orders):
            H = hermite(n)
            y = H(x)
            color = self.plot_manager.config.get_color(i)
            ax.plot(x, y, label=f"$H_{{{n}}}(x)$", color=color)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$H_n(x)$")
        ax.set_title("Hermite Polynomials")
        ax.legend()
        ax.grid(True, alpha=self.plot_manager.config.grid_alpha)

        return fig, ax
