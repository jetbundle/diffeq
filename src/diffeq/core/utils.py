"""Utility functions for common mathematical operations."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def create_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    num_points: int = 50,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Create 2D coordinate grids.

    Args:
        x_range: (x_min, x_max) tuple.
        y_range: (y_min, y_max) tuple.
        num_points: Number of points in each dimension.

    Returns:
        Tuple of (X, Y) meshgrid arrays.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


def create_grid_1d(
    x_range: Tuple[float, float],
    num_points: int = 100,
) -> NDArray[np.floating]:
    """Create 1D coordinate grid.

    Args:
        x_range: (x_min, x_max) tuple.
        num_points: Number of points.

    Returns:
        Array of x coordinates.
    """
    return np.linspace(x_range[0], x_range[1], num_points)


def normalize_vector_field(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Normalize a 2D vector field to unit vectors.

    Args:
        u: x-component of vector field.
        v: y-component of vector field.

    Returns:
        Tuple of normalized (u, v).
    """
    magnitude = np.sqrt(u ** 2 + v ** 2)
    magnitude[magnitude == 0] = 1.0
    return u / magnitude, v / magnitude
