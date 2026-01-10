"""Complex plane visualizations (for Stokes phenomenon, etc.)."""

from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.special import airy

from diffeq.core.plotting import PlotManager
from diffeq.core.config import PlotConfig, DEFAULT_PLOT_CONFIG


class ComplexPlaneVisualizer:
    """Visualize functions in the complex plane."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize complex plane visualizer.

        Args:
            config: Plot configuration. If None, uses default.
        """
        self.plot_manager = PlotManager(config=config or DEFAULT_PLOT_CONFIG)

    def plot_magnitude(
        self,
        func: Callable[[NDArray[np.complexfloating]], NDArray[np.complexfloating]],
        x_range: Tuple[float, float] = (-5.0, 5.0),
        y_range: Tuple[float, float] = (-5.0, 5.0),
        resolution: int = 400,
        colormap: str = "viridis",
        highlight_lines: Optional[list[float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the magnitude of a complex function.

        Args:
            func: Function f(z) that returns complex values.
            x_range: Real axis range.
            y_range: Imaginary axis range.
            resolution: Grid resolution (creates resolution x resolution grid).
            colormap: Colormap name for magnitude.
            highlight_lines: List of angles (in radians) for Stokes lines to highlight.

        Returns:
            Tuple of (figure, axes).
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Evaluate function
        F = func(Z)
        magnitude = np.abs(F)

        # Create plot
        fig, ax = self.plot_manager.create_figure()

        # Plot magnitude as colormap
        im = ax.contourf(X, Y, magnitude, levels=50, cmap=colormap)
        plt.colorbar(im, ax=ax, label="Magnitude")

        # Highlight Stokes lines if provided
        if highlight_lines:
            center = ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2)
            max_radius = np.sqrt((x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2) / 2

            for angle in highlight_lines:
                dx = max_radius * np.cos(angle)
                dy = max_radius * np.sin(angle)
                ax.plot(
                    [center[0] - dx, center[0] + dx],
                    [center[1] - dy, center[1] + dy],
                    "r--",
                    linewidth=self.plot_manager.config.line_width,
                    alpha=0.7,
                    label="Stokes Line" if angle == highlight_lines[0] else "",
                )

        ax.set_xlabel("$\\Re(z)$")
        ax.set_ylabel("$\\Im(z)$")
        ax.set_title("Complex Plane Magnitude")
        ax.set_aspect("equal")

        if highlight_lines:
            ax.legend()

        return fig, ax

    def plot_airy_magnitude(
        self,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        y_range: Tuple[float, float] = (-5.0, 5.0),
        resolution: int = 400,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot magnitude of Airy function with Stokes lines.

        Args:
            x_range: Real axis range.
            y_range: Imaginary axis range.
            resolution: Grid resolution.

        Returns:
            Tuple of (figure, axes).
        """
        def airy_func(z: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            """Compute Airy function Ai(z)."""
            result = np.zeros_like(z, dtype=complex)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    ai, _, _, _ = airy(z[i, j])
                    result[i, j] = ai
            return result

        # Stokes lines at angles: 0, 2π/3, 4π/3 (in complex plane, these are at arg(z) = ...)
        stokes_angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]

        return self.plot_magnitude(
            airy_func,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            highlight_lines=stokes_angles,
        )
