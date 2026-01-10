"""Configuration module for unified theming and sizing across all notebooks.

This module provides centralized configuration for:
- Dark theme plotting colors and styles
- Consistent figure sizing
- Widget layout and styling
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotConfig:
    """Centralized configuration for all plots with dark theme."""

    # Figure sizing (inches) - optimized for readability without being too large
    figure_width: float = 9.0
    figure_height: float = 6.5
    figure_dpi: int = 100

    # 3D plot sizing
    figure_3d_width: float = 10.0
    figure_3d_height: float = 8.0

    # Dark theme colors
    background_color: str = "#1e1e1e"
    plot_bg_color: str = "#252526"
    grid_color: str = "#3e3e42"
    text_color: str = "#cccccc"
    axis_color: str = "#858585"

    # Color palette for multiple series
    colors: Tuple[str, ...] = (
        "#4ec9b0",  # Teal - primary
        "#569cd6",  # Blue
        "#c586c0",  # Purple
        "#ce9178",  # Orange
        "#dcdcaa",  # Yellow
        "#9cdcfe",  # Light blue
        "#f48771",  # Red
        "#85c1e2",  # Sky blue
    )

    # Matplotlib style settings
    line_width: float = 2.0
    line_width_thick: float = 2.5
    line_width_thin: float = 1.5
    marker_size: float = 6.0
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.8

    # Font settings
    font_family: str = "serif"
    font_size: int = 11
    font_size_large: int = 13
    font_size_small: int = 9
    math_font: str = "dejavusans"

    # LaTeX rendering
    use_latex: bool = True

    def apply_style(self) -> None:
        """Apply this configuration to matplotlib's current style."""
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": self.background_color,
                "axes.facecolor": self.plot_bg_color,
                "axes.edgecolor": self.axis_color,
                "axes.labelcolor": self.text_color,
                "text.color": self.text_color,
                "xtick.color": self.axis_color,
                "ytick.color": self.axis_color,
                "grid.color": self.grid_color,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": self.grid_linewidth,
                "font.family": self.font_family,
                "font.size": self.font_size,
                "mathtext.fontset": self.math_font,
                "figure.dpi": self.figure_dpi,
            }
        )
        if self.use_latex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
                }
            )

    def get_color(self, index: int) -> str:
        """Get a color from the palette, cycling if necessary."""
        return self.colors[index % len(self.colors)]


@dataclass(frozen=True)
class WidgetConfig:
    """Configuration for ipywidgets styling and layout."""

    # Layout sizing
    widget_width: str = "100%"
    plot_height: str = "500px"
    plot_3d_height: str = "600px"

    # Widget styling
    description_width: str = "150px"
    button_width: str = "120px"
    slider_width: str = "300px"

    # Spacing
    margin_top: str = "10px"
    margin_bottom: str = "10px"
    padding: str = "10px"

    # Interactive update settings
    update_interval_ms: int = 100
    max_points: int = 10000

    @staticmethod
    def get_layout() -> dict:
        """Get a standard widget layout dictionary."""
        return {
            "width": WidgetConfig.widget_width,
            "margin": f"{WidgetConfig.margin_top} 0px {WidgetConfig.margin_bottom} 0px",
            "padding": WidgetConfig.padding,
        }


# Global default configuration instances
DEFAULT_PLOT_CONFIG = PlotConfig()
DEFAULT_WIDGET_CONFIG = WidgetConfig()
