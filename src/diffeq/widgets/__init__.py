"""Widget modules for interactive visualizations."""

from diffeq.widgets.base import InteractiveWidget, ParameterSlider
from diffeq.widgets.phase_portrait import PhasePortraitWidget
from diffeq.widgets.chaos_slider import ChaosSliderWidget

__all__ = [
    "InteractiveWidget",
    "ParameterSlider",
    "PhasePortraitWidget",
    "ChaosSliderWidget",
]
