"""Numerical solvers for ODEs and PDEs.

This module provides high-level interfaces to scipy.integrate and
custom numerical methods, with consistent error handling and
performance optimization.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from numpy.typing import NDArray


class ODESolver:
    """High-level interface for solving ODEs with scipy backend."""

    @staticmethod
    def solve_ivp(
        fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        t_span: Tuple[float, float],
        y0: NDArray[np.floating],
        t_eval: Optional[NDArray[np.floating]] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        dense_output: bool = False,
        **kwargs
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Solve an initial value problem for ODE system.

        Args:
            fun: Right-hand side of the system: dy/dt = fun(t, y).
            t_span: Interval of integration (t0, tf). Must satisfy t0 < tf.
            y0: Initial condition (shape: (n,)).
            t_eval: Times at which to store solution. If None, uses adaptive output.
            method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA').
            rtol: Relative tolerance (positive).
            atol: Absolute tolerance (positive).
            dense_output: Whether to compute a continuous solution.
            **kwargs: Additional arguments passed to solve_ivp.

        Returns:
            Tuple of (t, y) where y has shape (len(t), n).

        Raises:
            ValueError: If inputs are invalid or solution fails.
        """
        # Input validation
        y0_arr = np.asarray(y0)
        if y0_arr.ndim != 1:
            raise ValueError("y0 must be a 1D array")
        if y0_arr.size == 0:
            raise ValueError("y0 cannot be empty")

        t0, tf = float(t_span[0]), float(t_span[1])
        if t0 >= tf:
            raise ValueError("t_span[0] must be less than t_span[1]")

        if rtol <= 0:
            raise ValueError("rtol must be positive")
        if atol <= 0:
            raise ValueError("atol must be positive")

        if t_eval is not None:
            t_eval_arr = np.asarray(t_eval)
            if t_eval_arr.ndim != 1:
                raise ValueError("t_eval must be a 1D array")
            if len(t_eval_arr) > 0:
                if np.any(t_eval_arr < t0) or np.any(t_eval_arr > tf):
                    raise ValueError("t_eval values must be within t_span")

        result = solve_ivp(
            fun,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=dense_output,
            **kwargs
        )

        if not result.success:
            raise ValueError(f"ODE solver failed: {result.message}")

        return result.t, result.y.T

    @staticmethod
    def solve_odeint(
        fun: Callable[[NDArray[np.floating], float], NDArray[np.floating]],
        y0: NDArray[np.floating],
        t: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating]:
        """Solve ODE system using odeint (legacy interface).

        Args:
            fun: Right-hand side: dy/dt = fun(y, t).
            y0: Initial condition (shape: (n,)).
            t: Time points at which to solve (must be monotonically increasing).
            **kwargs: Additional arguments passed to odeint.

        Returns:
            Solution array with shape (len(t), n).

        Raises:
            ValueError: If inputs are invalid or solution fails.
        """
        # Input validation
        y0_arr = np.asarray(y0)
        t_arr = np.asarray(t)

        if y0_arr.ndim != 1:
            raise ValueError("y0 must be a 1D array")
        if y0_arr.size == 0:
            raise ValueError("y0 cannot be empty")

        if t_arr.ndim != 1:
            raise ValueError("t must be a 1D array")
        if t_arr.size == 0:
            raise ValueError("t cannot be empty")

        if len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0):
            raise ValueError("t must be monotonically increasing")

        sol = odeint(fun, y0, t, **kwargs)

        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            raise ValueError("ODE solution contains NaN or Inf values")

        return sol

    @staticmethod
    def lorenz_system(
        t: float,
        y: NDArray[np.floating],
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> NDArray[np.floating]:
        """Right-hand side of the Lorenz system.

        Args:
            t: Time (unused, required by solver interface).
            y: State vector [x, y, z].
            sigma: Prandtl number.
            rho: Rayleigh number.
            beta: Geometric parameter.

        Returns:
            Derivative vector [dx/dt, dy/dt, dz/dt].

        Raises:
            ValueError: If y does not have exactly 3 elements.
        """
        y_arr = np.asarray(y)
        if y_arr.size != 3:
            raise ValueError("Lorenz system requires state vector of length 3")

        x, y_val, z = y_arr[0], y_arr[1], y_arr[2]
        lorentz_system = np.array(
            [
                sigma * (y_val - x),
                x * (rho - z) - y_val,
                x * y_val - beta * z,
            ]
        )
        return lorenz_system

    @staticmethod
    def solve_lorenz(
        t_span: Tuple[float, float],
        y0: NDArray[np.floating],
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        num_points: int = 5000,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Solve the Lorenz system.

        Args:
            t_span: Time interval (t0, tf). Must satisfy t0 < tf.
            y0: Initial condition [x0, y0, z0].
            sigma: Prandtl number.
            rho: Rayleigh number.
            beta: Geometric parameter.
            num_points: Number of output points (positive integer).

        Returns:
            Tuple of (t, y) where y has shape (num_points, 3).

        Raises:
            ValueError: If inputs are invalid.
        """
        if num_points <= 0:
            raise ValueError("num_points must be a positive integer")

        y0_arr = np.asarray(y0)
        if y0_arr.size != 3:
            raise ValueError("Initial condition must have exactly 3 elements")

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        def rhs(t: float, y: NDArray[np.floating]) -> NDArray[np.floating]:
            return ODESolver.lorenz_system(t, y, sigma, rho, beta)

        solution_lorenz = ODESolver.solve_ivp(rhs, t_span, y0_arr, t_eval=t_eval, method="DOP853")

        return solution_lorenz

    @staticmethod
    def solve_duffing(
        t: NDArray[np.floating],
        y0: NDArray[np.floating],
        epsilon: float = 0.1,
    ) -> NDArray[np.floating]:
        """Solve the Duffing oscillator: d²x/dt² + x + εx³ = 0.

        Args:
            t: Time points (must be monotonically increasing).
            y0: Initial condition [x(0), dx/dt(0)].
            epsilon: Nonlinearity strength.

        Returns:
            Solution array with shape (len(t), 2) where columns are [x, dx/dt].

        Raises:
            ValueError: If inputs are invalid.
        """
        t_arr = np.asarray(t)
        y0_arr = np.asarray(y0)

        if t_arr.ndim != 1 or t_arr.size < 2:
            raise ValueError("t must be a 1D array with at least 2 points")

        if len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0):
            raise ValueError("t must be monotonically increasing")

        if y0_arr.size != 2:
            raise ValueError("Initial condition must have exactly 2 elements [x0, v0]")

        def rhs(tt: float, y: NDArray[np.floating]) -> NDArray[np.floating]:
            x, xdot = y[0], y[1]
            return np.array([xdot, -x - epsilon * x ** 3])

        t_span = (t_arr[0], t_arr[-1])

        solution_duffing = ODESolver.solve_ivp(rhs, t_span, y0_arr, t_eval=t_arr, method="DOP853")

        _, y = solution_duffing

        return y


class PDESolver:
    """High-level interface for solving PDEs (focus on 1D/2D problems)."""

    @staticmethod
    def solve_heat_1d(
        u0: NDArray[np.floating],
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        alpha: float = 1.0,
        bc_left: Tuple[str, float] = ("dirichlet", 0.0),
        bc_right: Tuple[str, float] = ("dirichlet", 0.0),
    ) -> NDArray[np.floating]:
        """Solve 1D heat equation: du/dt = alpha * d^2u/dx^2.

        Uses finite differences with implicit Euler (backward Euler) for stability.

        Args:
            u0: Initial condition u(x, 0) (shape: (nx,)).
            x: Spatial grid points (uniform spacing assumed).
            t: Time points (must be monotonically increasing).
            alpha: Diffusion coefficient (positive).
            bc_left: Left boundary condition (type, value).
                     Supported types: 'dirichlet', 'neumann'.
            bc_right: Right boundary condition (type, value).
                      Supported types: 'dirichlet', 'neumann'.

        Returns:
            Solution array with shape (len(t), len(x)).

        Raises:
            ValueError: If inputs are invalid or grid spacing is non-uniform.

        Notes:
            - Assumes regular/uniform spatial grid.
            - Uses implicit Euler method for unconditional stability.
        """
        # Input validation
        u0_arr = np.asarray(u0)
        x_arr = np.asarray(x)
        t_arr = np.asarray(t)
        
        if u0_arr.ndim != 1 or u0_arr.size < 2:
            raise ValueError("u0 must be a 1D array with at least 2 points")
        
        if x_arr.ndim != 1 or x_arr.size < 2:
            raise ValueError("x must be a 1D array with at least 2 points")
        
        if t_arr.ndim != 1 or t_arr.size < 1:
            raise ValueError("t must be a 1D array with at least 1 point")
        
        if u0_arr.size != x_arr.size:
            raise ValueError("u0 and x must have the same length")
        
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0):
            raise ValueError("t must be monotonically increasing")

        # Check uniform spacing
        dx_vals = np.diff(x_arr)
        if not np.allclose(dx_vals, dx_vals[0], rtol=1e-10):
            raise ValueError("x must have uniform spacing")

        # Validate boundary condition types
        valid_bc_types = {'dirichlet', 'neumann'}
        if bc_left[0] not in valid_bc_types:
            raise ValueError(f"bc_left type must be one of {valid_bc_types}")
        if bc_right[0] not in valid_bc_types:
            raise ValueError(f"bc_right type must be one of {valid_bc_types}")

        nx = len(x_arr)
        dx = dx_vals[0]

        if len(t_arr) == 1:
            # Single time point - return initial condition
            return u0_arr.reshape(1, -1)
 
        dt = t_arr[1] - t_arr[0]
        if dt <= 0:
            raise ValueError("Time step dt must be positive")

        r = alpha * dt / (dx ** 2)

        u = np.zeros((len(t_arr), nx))
        u[0, :] = u0_arr.copy()

        # Build tridiagonal matrix for implicit Euler
        diag_main = np.ones(nx) * (1 + 2 * r)
        diag_upper = np.ones(nx - 1) * (-r)
        diag_lower = np.ones(nx - 1) * (-r)

        # Apply boundary conditions
        if bc_left[0] == "dirichlet":
            diag_main[0] = 1.0
            diag_upper[0] = 0.0
        elif bc_left[0] == "neumann":
            diag_main[0] = 1.0 + r
            diag_upper[0] = -r

        if bc_right[0] == "dirichlet":
            diag_main[-1] = 1.0
            diag_lower[-1] = 0.0
        elif bc_right[0] == "neumann":
            diag_main[-1] = 1.0 + r
            diag_lower[-1] = -r

        A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csc")

        for i in range(1, len(t_arr)):
            b = u[i - 1, :].copy()
            if bc_left[0] == "dirichlet":
                b[0] = bc_left[1]
            if bc_right[0] == "dirichlet":
                b[-1] = bc_right[1]

            u[i, :] = spsolve(A, b)

        return u

    @staticmethod
    def solve_burgers_1d(
        u0: NDArray[np.floating],
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        nu: float = 0.01,
    ) -> NDArray[np.floating]:
        """Solve 1D Burgers equation: du/dt + u * du/dx = nu * d^2u/dx^2.

        Uses finite differences with upwind scheme for advection and
        implicit Euler for diffusion.

        Args:
            u0: Initial condition u(x, 0) (shape: (nx,)).
            x: Spatial grid points (uniform spacing assumed).
            t: Time points (must be monotonically increasing).
            nu: Viscosity coefficient (non-negative).

        Returns:
            Solution array with shape (len(t), len(x)).

        Raises:
            ValueError: If inputs are invalid or grid spacing is non-uniform.

        Notes:
            - Assumes regular/uniform spatial grid.
            - Uses operator splitting: explicit upwind for advection,
              implicit Euler for diffusion.
            - Dirichlet boundary conditions (u = 0 at boundaries).
        """
        # Input validation
        u0_arr = np.asarray(u0)
        x_arr = np.asarray(x)
        t_arr = np.asarray(t)

        if u0_arr.ndim != 1 or u0_arr.size < 2:
            raise ValueError("u0 must be a 1D array with at least 2 points")

        if x_arr.ndim != 1 or x_arr.size < 2:
            raise ValueError("x must be a 1D array with at least 2 points")

        if t_arr.ndim != 1 or t_arr.size < 1:
            raise ValueError("t must be a 1D array with at least 1 point")

        if u0_arr.size != x_arr.size:
            raise ValueError("u0 and x must have the same length")

        if nu < 0:
            raise ValueError("nu must be non-negative")

        if len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0):
            raise ValueError("t must be monotonically increasing")

        # Check uniform spacing
        dx_vals = np.diff(x_arr)
        if not np.allclose(dx_vals, dx_vals[0], rtol=1e-10):
            raise ValueError("x must have uniform spacing")

        nx = len(x_arr)
        dx = dx_vals[0]
        
        if len(t_arr) == 1:
            # Single time point - return initial condition
            return u0_arr.reshape(1, -1)
        
        dt = t_arr[1] - t_arr[0]
        if dt <= 0:
            raise ValueError("Time step dt must be positive")

        u = np.zeros((len(t_arr), nx))
        u[0, :] = u0_arr.copy()

        for i in range(1, len(t_arr)):
            u_prev = u[i - 1, :]

            # Upwind finite difference for advection
            ux = np.zeros(nx)
            ux[1:] = (u_prev[1:] - u_prev[:-1]) / dx
            ux[0] = ux[1]  # Extrapolate at boundary

            # Diffusion term (implicit)
            r = nu * dt / (dx ** 2)
            diag_main = np.ones(nx) * (1 + 2 * r)
            diag_upper = np.ones(nx - 1) * (-r)
            diag_lower = np.ones(nx - 1) * (-r)

            # Dirichlet boundary conditions
            diag_main[0] = 1.0
            diag_main[-1] = 1.0
            diag_upper[0] = 0.0
            diag_lower[-1] = 0.0

            A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csc")

            # Nonlinear advection term (explicit)
            b = u_prev - dt * u_prev * ux

            # Enforce boundary conditions
            b[0] = 0.0
            b[-1] = 0.0

            u[i, :] = spsolve(A, b)

        return u
