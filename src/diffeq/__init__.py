"""Differential Equations Visualization and Analysis Package.

A backend for interactive mathematical notebooks exploring
differential equations through seven conceptual levels.
"""

__version__ = "0.1.0"

from diffeq.core.config import PlotConfig, WidgetConfig
from diffeq.core.plotting import PlotManager
from diffeq.core.solvers import ODESolver, PDESolver

__all__ = [
    "PlotConfig",
    "WidgetConfig",
    "PlotManager",
    "ODESolver",
    "PDESolver",
]
