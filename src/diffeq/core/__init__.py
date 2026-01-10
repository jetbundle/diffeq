"""Core modules for differential equation visualization and analysis."""

from diffeq.core.config import PlotConfig, WidgetConfig
from diffeq.core.plotting import PlotManager
from diffeq.core.solvers import ODESolver, PDESolver
from diffeq.core.utils import create_grid_2d, create_grid_1d, normalize_vector_field

__all__ = [
    "PlotConfig",
    "WidgetConfig",
    "PlotManager",
    "ODESolver",
    "PDESolver",
    "create_grid_2d",
    "create_grid_1d",
    "normalize_vector_field",
]
