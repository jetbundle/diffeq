"""Base classes for interactive widgets.

This module provides base classes and utilities for creating
consistent, reusable interactive widgets using ipywidgets.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from numpy.typing import NDArray

from diffeq.core.config import WidgetConfig, DEFAULT_WIDGET_CONFIG
from diffeq.core.plotting import PlotManager


class ParameterSlider:
    """Helper class for creating parameter sliders with consistent styling."""

    def __init__(
        self,
        name: str,
        value: float,
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        description: Optional[str] = None,
        config: Optional[WidgetConfig] = None,
    ):
        """Create a parameter slider.

        Args:
            name: Parameter name (used as key in value dictionaries).
            value: Initial value.
            min_val: Minimum value.
            max_val: Maximum value.
            step: Step size. If None, auto-calculated.
            description: Display description. If None, uses name.
            config: Widget configuration. If None, uses default.
        """
        self.name = name
        self.config = config or DEFAULT_WIDGET_CONFIG

        if step is None:
            step = (max_val - min_val) / 100.0

        if description is None:
            description = name

        self.widget = widgets.FloatSlider(
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            description=description,
            style={"description_width": self.config.description_width},
            layout=widgets.Layout(width=self.config.slider_width),
        )

    def get_value(self) -> float:
        """Get current slider value."""
        return self.widget.value


class InteractiveWidget(ABC):
    """Abstract base class for interactive widgets.

    Provides a template for creating widgets with:
    - Parameter sliders
    - Output display area
    - Update mechanism
    - Consistent layout
    """

    def __init__(
        self,
        title: Optional[str] = None,
        config: Optional[WidgetConfig] = None,
        plot_config: Optional[Any] = None,
    ):
        """Initialize the interactive widget.

        Args:
            title: Widget title (displayed at top).
            config: Widget configuration. If None, uses default.
            plot_config: Plot configuration. If None, uses default.
        """
        self.config = config or DEFAULT_WIDGET_CONFIG
        self.plot_manager = PlotManager(config=plot_config)
        self.title = title

        self.sliders: Dict[str, ParameterSlider] = {}
        self.output = widgets.Output(layout={"height": self.config.plot_height})

        self._setup_widgets()
        self._setup_observers()

    @abstractmethod
    def _setup_widgets(self) -> None:
        """Set up the widget UI components (sliders, buttons, etc.).

        Subclasses must implement this to create their specific UI.
        """
        pass

    @abstractmethod
    def _update_plot(self, change: Optional[Any] = None) -> None:
        """Update the plot when parameters change.

        Args:
            change: Widget change event (from ipywidgets observe).
        """
        pass

    def _setup_observers(self) -> None:
        """Set up observers for parameter changes."""
        for slider in self.sliders.values():
            slider.widget.observe(self._update_plot, names="value")

    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values.

        Returns:
            Dictionary mapping parameter names to current values.
        """
        return {name: slider.get_value() for name, slider in self.sliders.items()}

    def _create_layout(self) -> widgets.VBox:
        """Create the widget layout.

        Returns:
            VBox containing all widget components.
        """
        children: List[widgets.Widget] = []

        if self.title:
            title_widget = widgets.HTML(
                value=f"<h3>{self.title}</h3>",
                layout=WidgetConfig.get_layout(),
            )
            children.append(title_widget)

        if self.sliders:
            slider_box = widgets.HBox(
                [slider.widget for slider in self.sliders.values()],
                layout=widgets.Layout(
                    flex_flow="row wrap",
                    justify_content="flex-start",
                    margin=self.config.margin_top,
                ),
            )
            children.append(slider_box)

        children.append(self.output)

        return widgets.VBox(children, layout=WidgetConfig.get_layout())

    def display(self) -> widgets.VBox:
        """Display the widget.

        Returns:
            The widget container (VBox).
        """
        return self._create_layout()


class AnimationWidget(InteractiveWidget):
    """Base class for widgets with animation capability."""

    def __init__(
        self,
        title: Optional[str] = None,
        config: Optional[WidgetConfig] = None,
        plot_config: Optional[Any] = None,
        auto_play: bool = False,
        fps: int = 30,
    ):
        """Initialize animation widget.

        Args:
            title: Widget title.
            config: Widget configuration.
            plot_config: Plot configuration.
            auto_play: Whether to start animation automatically.
            fps: Frames per second for animation.
        """
        super().__init__(title, config, plot_config)
        self.auto_play = auto_play
        self.fps = fps
        self.is_playing = False
        self.play_button: Optional[widgets.Button] = None
        self.time_slider: Optional[ParameterSlider] = None

    def _setup_animation_controls(self) -> None:
        """Set up animation control buttons."""
        self.play_button = widgets.Button(
            description="Play" if not self.is_playing else "Pause",
            button_style="",
            layout=widgets.Layout(width=self.config.button_width),
        )
        self.play_button.on_click(self._toggle_play)

    def _toggle_play(self, button: widgets.Button) -> None:
        """Toggle play/pause state."""
        self.is_playing = not self.is_playing
        button.description = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            self._animate()

    @abstractmethod
    def _animate(self) -> None:
        """Animation loop (to be implemented by subclasses)."""
        pass
