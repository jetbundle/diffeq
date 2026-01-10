"""Lorenz system chaos slider widget."""

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from diffeq.widgets.base import InteractiveWidget, ParameterSlider
from diffeq.core.config import WidgetConfig, PlotConfig
from diffeq.core.plotting import PlotManager
from diffeq.core.solvers import ODESolver


class ChaosSliderWidget(InteractiveWidget):
    """Interactive widget for Lorenz attractor with rho parameter slider."""

    def __init__(
        self,
        title: Optional[str] = None,
        config: Optional[WidgetConfig] = None,
        plot_config: Optional[PlotConfig] = None,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
    ):
        """Initialize chaos slider widget.

        Args:
            title: Widget title.
            config: Widget configuration.
            plot_config: Plot configuration.
            sigma: Prandtl number (fixed).
            beta: Geometric parameter (fixed).
        """
        self.sigma = sigma
        self.beta = beta

        super().__init__(title=title or "Lorenz System: The Chaos Slider", config=config, plot_config=plot_config)

    def _setup_widgets(self) -> None:
        """Set up chaos slider UI."""
        self.sliders["rho"] = ParameterSlider(
            name="rho",
            value=28.0,
            min_val=0.0,
            max_val=50.0,
            step=0.5,
            description=r"$\rho$ (Rayleigh)",
            config=self.config,
        )

    def _update_plot(self, change: Optional[Any] = None) -> None:
        """Update the Lorenz attractor plot."""
        with self.output:
            self.output.clear_output(wait=True)

            params = self.get_parameters()
            rho = params["rho"]

            # Initial condition
            y0 = np.array([1.0, 1.0, 1.0])

            # Solve Lorenz system
            t_span = (0.0, 50.0)
            t, y = ODESolver.solve_lorenz(
                t_span=t_span,
                y0=y0,
                sigma=self.sigma,
                rho=rho,
                beta=self.beta,
                num_points=5000,
            )

            # Create 3D plot
            fig, ax = self.plot_manager.create_figure(projection="3d")

            # Plot trajectory with color gradient
            ax.plot(y[:, 0], y[:, 1], y[:, 2], linewidth=self.plot_manager.config.line_width_thin, alpha=0.8)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_title(f"Lorenz Attractor ($\\rho = {rho:.1f}$)")

            plt.tight_layout()
            plt.show()
