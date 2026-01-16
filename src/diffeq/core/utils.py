"""Utility functions for common mathematical operations."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def create_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    num_points: int = 50,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Create 2D coordinate grids (meshgrid) for the given ranges.

    Args:
    x_range: (x_min, x_max) tuple. x_min must be < x_max.
    y_range: (y_min, y_max) tuple. y_min must be < y_max.
    num_points: Number of points in each dimension (positive integer).
    
    
    Returns:
    Tuple of (X, Y) meshgrid arrays, each with shape (num_points, num_points).

    Notes:
    The returned arrays use the conventional "xy" indexing so that X varies
    along the second axis and Y varies along the first axis 
    (i.e. both are compatible with `plt.contourf(X, Y, Z)` usage).
    """
    if num_points <= 0:
    raise ValueError("num_points must be a positive integer")

    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])

    if x_min >= x_max:
        raise ValueError("x_range[0] must be less than x_range[1]")
    if y_min >= y_max:
        raise ValueError("y_range[0] must be less than y_range[1]")

    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y

def create_grid_1d(
    x_range: Tuple[float, float],
    num_points: int = 100,
) -> NDArray[np.floating]:
    """Create 1D coordinate grid.

    Args:
        x_range: (x_min, x_max) tuple. x_min must be < x_max.
        num_points: Number of points (positive integer).

    Returns:
        1D array of x coordinates with length `num_points`.
    """
    if num_points <= 0:
        raise ValueError("num_points must be a positive integer")
    
    
    x_min, x_max = float(x_range[0]), float(x_range[1])
    if x_min >= x_max:
        raise ValueError("x_range[0] must be less than x_range[1]")
    return np.linspace(x_range[0], x_range[1], num_points)


def normalize_vector_field(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Normalize a 2D vector field to unit vectors.

    Args:
        u: x-component of vector field (array-like).
        v: y-component of vector field (array-like).

    Returns:
        Tuple of normalized (u_norm, v_norm), each an array of floating dtype
        and the same shape as the inputs.
    
    Behavior:
        - If an input location has magnitude zero, the returned vector at that
        location is (0.0, 0.0).
        - Input integer arrays are safely promoted to floating dtype for the
        computation so the outputs are floating-point arrays (avoids silent
        integer truncation on division).
    """
    u_arr = np.asarray(u)
    v_arr = np.asarray(v)

    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape")

    dtype_out = np.result_type(u_arr.dtype, v_arr.dtype, np.float64)
    u_f = u_arr.astype(dtype_out, copy=False)
    v_f = v_arr.astype(dtype_out, copy=False)

    magnitude = np.hypot(u_f, v_f)

    u_norm = np.empty_like(u_f)
    v_norm = np.empty_like(v_f)

    nonzero = magnitude != 0
    np.divide(u_f, magnitude, out=u_norm, where=nonzero)
    np.divide(v_f, magnitude, out=v_norm, where=nonzero)

    if not np.all(nonzero):
        u_norm[~nonzero] = dtype_out.type(0.0)
        v_norm[~nonzero] = dtype_out.type(0.0)

    return u_norm, v_norm
