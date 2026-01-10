"""Unified plotting interface for matplotlib and plotly.

This module provides a high-level interface for creating consistent,
dark-themed plots across all notebooks. It handles both static matplotlib
plots and interactive plotly visualizations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from diffeq.core.config import PlotConfig, DEFAULT_PLOT_CONFIG


class PlotManager:
    """Manages plotting with unified dark theme and consistent styling.

    This class provides a high-level interface for creating plots with
    automatic dark theme application and consistent sizing.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize the plot manager.

        Args:
            config: Plot configuration. If None, uses DEFAULT_PLOT_CONFIG.
        """
        self.config = config or DEFAULT_PLOT_CONFIG
        self.config.apply_style()

    def create_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        projection: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[Figure, Axes]:
        """Create a matplotlib figure and axes with configured styling.

        Args:
            figsize: Figure size (width, height) in inches. If None, uses config defaults.
            projection: Projection type (e.g., '3d', 'polar').
            **kwargs: Additional arguments passed to plt.subplots.

        Returns:
            Tuple of (figure, axes).
        """
        if figsize is None:
            if projection == "3d":
                figsize = (self.config.figure_3d_width, self.config.figure_3d_height)
            else:
                figsize = (self.config.figure_width, self.config.figure_height)

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.config.background_color, **kwargs)

        if projection == "3d":
            ax = fig.add_subplot(111, projection="3d")

        ax.set_facecolor(self.config.plot_bg_color)
        ax.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)

        return fig, ax

    def plot_2d(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        label: Optional[str] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        linestyle: str = "-",
        marker: Optional[str] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any
    ) -> Axes:
        """Plot a 2D line with consistent styling.

        Args:
            x: x-coordinates.
            y: y-coordinates.
            label: Line label for legend.
            color: Line color. If None, cycles through palette.
            linewidth: Line width. If None, uses config default.
            linestyle: Line style ('-', '--', ':', etc.).
            marker: Marker style (None for lines only).
            ax: Axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.plot.

        Returns:
            The axes object.
        """
        if ax is None:
            _, ax = self.create_figure()

        if linewidth is None:
            linewidth = self.config.line_width

        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=self.config.marker_size if marker else None,
            **kwargs
        )

        if label:
            ax.legend(facecolor=self.config.plot_bg_color, edgecolor=self.config.axis_color)

        ax.set_xlabel(ax.get_xlabel() or "", color=self.config.text_color)
        ax.set_ylabel(ax.get_ylabel() or "", color=self.config.text_color)

        return ax

    def plot_vector_field(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        density: float = 1.0,
        color: Optional[str] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any
    ) -> Axes:
        """Plot a 2D vector field (quiver plot).

        Args:
            x: x-coordinate grid.
            y: y-coordinate grid.
            u: x-component of vectors.
            v: y-component of vectors.
            density: Controls density of arrows (1.0 = default).
            color: Arrow color. If None, uses default.
            ax: Axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.quiver.

        Returns:
            The axes object.
        """
        if ax is None:
            _, ax = self.create_figure()

        if color is None:
            color = self.config.colors[0]

        ax.quiver(
            x,
            y,
            u,
            v,
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1.0 / density,
            width=0.003,
            **kwargs
        )

        return ax

    def plot_3d_surface(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        color: Optional[str] = None,
        colormap: str = "viridis",
        ax: Optional[Axes] = None,
        **kwargs: Any
    ) -> Axes:
        """Plot a 3D surface.

        Args:
            x: x-coordinate grid.
            y: y-coordinate grid.
            z: z-coordinates (height).
            color: Surface color (single color).
            colormap: Colormap name (if color is None).
            ax: 3D axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.plot_surface.

        Returns:
            The 3D axes object.
        """
        if ax is None:
            _, ax = self.create_figure(projection="3d")

        if color:
            ax.plot_surface(x, y, z, color=color, **kwargs)
        else:
            ax.plot_surface(x, y, z, cmap=colormap, **kwargs)

        ax.set_xlabel(ax.get_xlabel() or "", color=self.config.text_color)
        ax.set_ylabel(ax.get_ylabel() or "", color=self.config.text_color)
        ax.set_zlabel(ax.get_zlabel() or "", color=self.config.text_color)

        return ax

    def plot_3d_trajectory(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any
    ) -> Axes:
        """Plot a 3D trajectory (parametric curve).

        Args:
            x: x-coordinates.
            y: y-coordinates.
            z: z-coordinates.
            color: Line color. If None, uses default.
            linewidth: Line width. If None, uses config default.
            ax: 3D axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to ax.plot.

        Returns:
            The 3D axes object.
        """
        if ax is None:
            _, ax = self.create_figure(projection="3d")

        if color is None:
            color = self.config.colors[0]

        if linewidth is None:
            linewidth = self.config.line_width

        ax.plot(x, y, z, color=color, linewidth=linewidth, **kwargs)

        ax.set_xlabel(ax.get_xlabel() or "", color=self.config.text_color)
        ax.set_ylabel(ax.get_ylabel() or "", color=self.config.text_color)
        ax.set_zlabel(ax.get_zlabel() or "", color=self.config.text_color)

        return ax

    def create_plotly_figure(
        self,
        layout: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any
    ) -> go.Figure:
        """Create a plotly figure with dark theme.

        Args:
            layout: Custom layout dictionary (merged with default dark theme).
            width: Figure width in pixels. If None, uses config default.
            height: Figure height in pixels. If None, uses config default.
            **kwargs: Additional arguments passed to go.Figure.

        Returns:
            Plotly figure with dark theme applied.
        """
        if width is None:
            width = int(self.config.figure_width * self.config.figure_dpi)
        if height is None:
            height = int(self.config.figure_height * self.config.figure_dpi)

        dark_layout = {
            "template": "plotly_dark",
            "paper_bgcolor": self.config.background_color,
            "plot_bgcolor": self.config.plot_bg_color,
            "font": {
                "family": self.config.font_family,
                "size": self.config.font_size,
                "color": self.config.text_color,
            },
            "xaxis": {
                "gridcolor": self.config.grid_color,
                "gridwidth": self.config.grid_linewidth,
                "showgrid": True,
            },
            "yaxis": {
                "gridcolor": self.config.grid_color,
                "gridwidth": self.config.grid_linewidth,
                "showgrid": True,
            },
            "width": width,
            "height": height,
        }

        if layout:
            dark_layout.update(layout)

        fig = go.Figure(**kwargs)
        fig.update_layout(dark_layout)
        return fig

    def plotly_line(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        name: Optional[str] = None,
        color: Optional[str] = None,
        line_width: Optional[float] = None,
        line_dash: Optional[str] = None,
        fig: Optional[go.Figure] = None,
        **kwargs: Any
    ) -> go.Figure:
        """Add a line trace to a Plotly figure.

        Args:
            x: x-coordinates.
            y: y-coordinates.
            name: Trace name for legend.
            color: Line color. If None, uses default.
            line_width: Line width. If None, uses config default.
            line_dash: Line dash style ('solid', 'dash', 'dot', 'dashdot').
            fig: Existing figure. If None, creates new one.
            **kwargs: Additional arguments passed to go.Scatter.

        Returns:
            Plotly figure.
        """
        if fig is None:
            fig = self.create_plotly_figure()

        if color is None:
            color = self.config.colors[0]

        if line_width is None:
            line_width = self.config.line_width

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=line_width, dash=line_dash),
                **kwargs
            )
        )

        return fig

    def plotly_3d_line(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        name: Optional[str] = None,
        color: Optional[str] = None,
        line_width: Optional[float] = None,
        fig: Optional[go.Figure] = None,
        **kwargs: Any
    ) -> go.Figure:
        """Add a 3D line trace to a Plotly figure.

        Args:
            x: x-coordinates.
            y: y-coordinates.
            z: z-coordinates.
            name: Trace name for legend.
            color: Line color. If None, uses default.
            line_width: Line width. If None, uses config default.
            fig: Existing figure. If None, creates new one.
            **kwargs: Additional arguments passed to go.Scatter3d.

        Returns:
            Plotly figure.
        """
        if fig is None:
            fig = self.create_plotly_figure()

        if color is None:
            color = self.config.colors[0]

        if line_width is None:
            line_width = self.config.line_width

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=name,
                line=dict(color=color, width=line_width),
                **kwargs
            )
        )

        return fig

    @staticmethod
    def show() -> None:
        """Display all pending matplotlib figures."""
        plt.tight_layout()
        plt.show()
