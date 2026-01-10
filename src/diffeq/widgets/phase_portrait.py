"""Phase portrait visualization widget."""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from diffeq.widgets.base import InteractiveWidget, ParameterSlider
from diffeq.core.config import WidgetConfig, PlotConfig
from diffeq.core.plotting import PlotManager


class PhasePortraitWidget(InteractiveWidget):
    """Interactive widget for visualizing 2D phase portraits with vector fields."""

    def __init__(
        self,
        system: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        x_range: Tuple[float, float] = (-5.0, 5.0),
        y_range: Tuple[float, float] = (-5.0, 5.0),
        grid_size: int = 20,
        title: Optional[str] = None,
        config: Optional[WidgetConfig] = None,
        plot_config: Optional[PlotConfig] = None,
    ):
        """Initialize phase portrait widget.

        Args:
            system: Function f(y) where dy/dt = f(y), y is [x, y].
            x_range: x-axis range.
            y_range: y-axis range.
            grid_size: Number of grid points for vector field.
            title: Widget title.
            config: Widget configuration.
            plot_config: Plot configuration.
        """
        self.system = system
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size

        super().__init__(title=title or "Phase Portrait", config=config, plot_config=plot_config)

    def _setup_widgets(self) -> None:
        """Set up phase portrait UI."""
        pass

    def _update_plot(self, change: Optional[Any] = None) -> None:
        """Update the phase portrait plot."""
        with self.output:
            self.output.clear_output(wait=True)

            fig, ax = self.plot_manager.create_figure()

            # Create grid for vector field
            x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
            y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
            X, Y = np.meshgrid(x, y)

            # Compute vector field
            U = np.zeros_like(X)
            V = np.zeros_like(Y)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    y_vec = np.array([X[i, j], Y[i, j]])
                    dydt = self.system(y_vec)
                    U[i, j] = dydt[0]
                    V[i, j] = dydt[1]

            # Normalize vectors for better visualization
            U, V = normalize_vector_field(U, V)

            self.plot_manager.plot_vector_field(X, Y, U, V, density=1.2, ax=ax)

            ax.set_xlim(self.x_range)
            ax.set_ylim(self.y_range)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title(self.title or "Phase Portrait")

            plt.tight_layout()
            plt.show()
