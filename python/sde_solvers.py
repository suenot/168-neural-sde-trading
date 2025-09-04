"""
SDE Solvers
===========

Numerical solvers for Stochastic Differential Equations.
Implements Euler-Maruyama, Milstein, and Stochastic Runge-Kutta methods.

These can be used standalone or as fallbacks when torchsde is not available.

All solvers handle the general SDE:
    dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)
"""

import math
from typing import Callable, Optional, Tuple

import torch


def euler_maruyama(
    drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 0.01,
    return_all_steps: bool = False,
) -> torch.Tensor:
    """
    Euler-Maruyama SDE solver.

    X(t+dt) = X(t) + f(X,t)*dt + g(X,t)*dW
    where dW ~ N(0, dt)

    Strong order: 0.5
    Weak order: 1.0

    Args:
        drift_fn: f(x, t) -> drift vector
        diffusion_fn: g(x, t) -> diffusion coefficient
        x0: Initial state (batch, dim)
        ts: Evaluation time points (num_times,)
        dt: Step size
        return_all_steps: If True, return at every solver step (not just ts)

    Returns:
        Trajectory tensor at evaluation points (num_times, batch, dim)
    """
    trajectory = [x0]
    x = x0.clone()
    t_current = ts[0].item()
    time_idx = 1

    all_steps = [x0] if return_all_steps else None

    while time_idx < len(ts):
        t_target = ts[time_idx].item()

        while t_current < t_target - 1e-10:
            actual_dt = min(dt, t_target - t_current)
            t_tensor = torch.tensor(t_current, dtype=x.dtype, device=x.device)

            # Compute drift and diffusion
            f = drift_fn(x, t_tensor)
            g = diffusion_fn(x, t_tensor)

            # Brownian increment
            dW = torch.randn_like(x) * math.sqrt(actual_dt)

            # Euler-Maruyama step
            x = x + f * actual_dt + g * dW

            t_current += actual_dt

            if return_all_steps and all_steps is not None:
                all_steps.append(x.clone())

        trajectory.append(x.clone())
        time_idx += 1

    if return_all_steps and all_steps is not None:
        return torch.stack(all_steps)

    return torch.stack(trajectory)


def milstein(
    drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 0.01,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Milstein SDE solver.

    X(t+dt) = X(t) + f(X,t)*dt + g(X,t)*dW + 0.5*g(X,t)*g'(X,t)*(dW^2 - dt)

    where g'(X,t) = dg/dX is computed via finite differences.

    Strong order: 1.0
    Weak order: 1.0

    Args:
        drift_fn: f(x, t) -> drift vector
        diffusion_fn: g(x, t) -> diffusion coefficient
        x0: Initial state (batch, dim)
        ts: Evaluation time points (num_times,)
        dt: Step size
        eps: Finite difference step for computing dg/dx

    Returns:
        Trajectory (num_times, batch, dim)
    """
    trajectory = [x0]
    x = x0.clone()
    t_current = ts[0].item()
    time_idx = 1

    while time_idx < len(ts):
        t_target = ts[time_idx].item()

        while t_current < t_target - 1e-10:
            actual_dt = min(dt, t_target - t_current)
            t_tensor = torch.tensor(t_current, dtype=x.dtype, device=x.device)

            # Compute drift and diffusion
            f = drift_fn(x, t_tensor)
            g = diffusion_fn(x, t_tensor)

            # Approximate dg/dx via finite differences
            g_plus = diffusion_fn(x + eps, t_tensor)
            g_prime = (g_plus - g) / eps

            # Brownian increment
            dW = torch.randn_like(x) * math.sqrt(actual_dt)

            # Milstein step
            x = (
                x
                + f * actual_dt
                + g * dW
                + 0.5 * g * g_prime * (dW ** 2 - actual_dt)
            )

            t_current += actual_dt

        trajectory.append(x.clone())
        time_idx += 1

    return torch.stack(trajectory)


def stochastic_runge_kutta(
    drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Stochastic Runge-Kutta (SRK) method.

    A two-stage method that achieves strong order 1.0 without
    requiring derivatives of the diffusion coefficient.

    Stage 1: K1 = f(X,t)*dt + g(X,t)*dW
    Supporting value: X_hat = X + K1
    Stage 2: K2 = f(X_hat, t+dt)*dt + g(X_hat, t+dt)*dW

    X(t+dt) = X(t) + 0.5*(K1 + K2)

    Args:
        drift_fn: f(x, t) -> drift vector
        diffusion_fn: g(x, t) -> diffusion coefficient
        x0: Initial state (batch, dim)
        ts: Evaluation time points (num_times,)
        dt: Step size

    Returns:
        Trajectory (num_times, batch, dim)
    """
    trajectory = [x0]
    x = x0.clone()
    t_current = ts[0].item()
    time_idx = 1

    while time_idx < len(ts):
        t_target = ts[time_idx].item()

        while t_current < t_target - 1e-10:
            actual_dt = min(dt, t_target - t_current)
            t_tensor = torch.tensor(t_current, dtype=x.dtype, device=x.device)
            t_next = torch.tensor(
                t_current + actual_dt, dtype=x.dtype, device=x.device
            )

            # Brownian increment (shared between stages)
            dW = torch.randn_like(x) * math.sqrt(actual_dt)

            # Stage 1
            f1 = drift_fn(x, t_tensor)
            g1 = diffusion_fn(x, t_tensor)
            K1 = f1 * actual_dt + g1 * dW

            # Supporting value
            x_hat = x + K1

            # Stage 2
            f2 = drift_fn(x_hat, t_next)
            g2 = diffusion_fn(x_hat, t_next)
            K2 = f2 * actual_dt + g2 * dW

            # SRK update
            x = x + 0.5 * (K1 + K2)

            t_current += actual_dt

        trajectory.append(x.clone())
        time_idx += 1

    return torch.stack(trajectory)


def adaptive_euler_maruyama(
    drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt_init: float = 0.01,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    dt_min: float = 1e-6,
    dt_max: float = 0.1,
    safety_factor: float = 0.9,
) -> torch.Tensor:
    """
    Adaptive step-size Euler-Maruyama solver.

    Uses step-doubling to estimate the local error and adjust
    the step size accordingly.

    Args:
        drift_fn: f(x, t) -> drift vector
        diffusion_fn: g(x, t) -> diffusion coefficient
        x0: Initial state (batch, dim)
        ts: Evaluation time points (num_times,)
        dt_init: Initial step size
        atol: Absolute error tolerance
        rtol: Relative error tolerance
        dt_min: Minimum step size
        dt_max: Maximum step size
        safety_factor: Safety factor for step size control

    Returns:
        Trajectory (num_times, batch, dim)
    """
    trajectory = [x0]
    x = x0.clone()
    t_current = ts[0].item()
    time_idx = 1
    dt = dt_init

    while time_idx < len(ts):
        t_target = ts[time_idx].item()

        while t_current < t_target - 1e-10:
            actual_dt = min(dt, t_target - t_current)
            t_tensor = torch.tensor(t_current, dtype=x.dtype, device=x.device)

            # Shared Brownian motion
            dW = torch.randn_like(x) * math.sqrt(actual_dt)

            # Full step
            f = drift_fn(x, t_tensor)
            g = diffusion_fn(x, t_tensor)
            x_full = x + f * actual_dt + g * dW

            # Two half steps
            half_dt = actual_dt / 2
            dW1 = dW * math.sqrt(0.5)  # Approximate split
            dW2 = (dW - dW1 * math.sqrt(0.5)) / math.sqrt(0.5)

            f1 = drift_fn(x, t_tensor)
            g1 = diffusion_fn(x, t_tensor)
            x_half1 = x + f1 * half_dt + g1 * dW1 * math.sqrt(half_dt / actual_dt)

            t_mid = torch.tensor(
                t_current + half_dt, dtype=x.dtype, device=x.device
            )
            f2 = drift_fn(x_half1, t_mid)
            g2 = diffusion_fn(x_half1, t_mid)
            x_half2 = (
                x_half1
                + f2 * half_dt
                + g2 * dW2 * math.sqrt(half_dt / actual_dt)
            )

            # Error estimate
            error = (x_full - x_half2).abs()
            tol = atol + rtol * x.abs().clamp(min=1e-10)
            ratio = (error / tol).max().item()

            if ratio <= 1.0 or actual_dt <= dt_min:
                # Accept step (use the more accurate half-step result)
                x = x_half2
                t_current += actual_dt

                # Increase step size
                if ratio > 0:
                    dt = min(
                        dt_max,
                        safety_factor * actual_dt * (1.0 / ratio) ** 0.5,
                    )
                else:
                    dt = min(dt_max, actual_dt * 2.0)
            else:
                # Reject step, reduce step size
                dt = max(
                    dt_min,
                    safety_factor * actual_dt * (1.0 / ratio) ** 0.5,
                )

        trajectory.append(x.clone())
        time_idx += 1

    return torch.stack(trajectory)


def geometric_brownian_motion(
    mu: float,
    sigma: float,
    s0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 0.001,
) -> torch.Tensor:
    """
    Exact simulation of Geometric Brownian Motion (baseline).

    dS = mu*S*dt + sigma*S*dW

    Uses the exact solution: S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))

    Args:
        mu: Drift parameter
        sigma: Volatility parameter
        s0: Initial price(s) (batch,)
        ts: Time points (num_times,)
        dt: Not used (exact solution), kept for API consistency

    Returns:
        Price paths (num_times, batch)
    """
    if s0.dim() == 0:
        s0 = s0.unsqueeze(0)

    num_times = len(ts)
    batch = s0.shape[0]

    # Compute time increments
    dt_vec = ts[1:] - ts[:-1]

    # Generate Brownian increments
    dW = torch.randn(num_times - 1, batch, device=s0.device)
    dW = dW * dt_vec.unsqueeze(-1).sqrt()

    # Cumulative Brownian motion
    W = torch.cat([torch.zeros(1, batch, device=s0.device), dW.cumsum(dim=0)])

    # Exact GBM solution
    t_expanded = ts.unsqueeze(-1)
    paths = s0.unsqueeze(0) * torch.exp(
        (mu - 0.5 * sigma ** 2) * t_expanded + sigma * W
    )

    return paths


def solve_sde(
    drift_fn: Callable,
    diffusion_fn: Callable,
    x0: torch.Tensor,
    ts: torch.Tensor,
    method: str = "euler",
    dt: float = 0.01,
    **kwargs,
) -> torch.Tensor:
    """
    Unified SDE solver interface.

    Args:
        drift_fn: Drift function f(x, t)
        diffusion_fn: Diffusion function g(x, t)
        x0: Initial state
        ts: Time points
        method: 'euler', 'milstein', 'srk', or 'adaptive'
        dt: Step size
        **kwargs: Additional solver-specific arguments

    Returns:
        Solution trajectory
    """
    solvers = {
        "euler": euler_maruyama,
        "milstein": milstein,
        "srk": stochastic_runge_kutta,
        "adaptive": adaptive_euler_maruyama,
    }

    if method not in solvers:
        raise ValueError(
            f"Unknown solver '{method}'. Choose from: {list(solvers.keys())}"
        )

    return solvers[method](drift_fn, diffusion_fn, x0, ts, dt=dt, **kwargs)


def compare_solvers(
    drift_fn: Callable,
    diffusion_fn: Callable,
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 0.01,
    num_trials: int = 100,
) -> dict:
    """
    Compare different SDE solvers on the same problem.

    Runs multiple trials with shared Brownian motions (where possible)
    and reports statistics.

    Args:
        drift_fn: Drift function
        diffusion_fn: Diffusion function
        x0: Initial state
        ts: Time points
        dt: Step size
        num_trials: Number of trials

    Returns:
        Dictionary with solver statistics
    """
    import time

    results = {}

    for name, solver in [
        ("euler", euler_maruyama),
        ("milstein", milstein),
        ("srk", stochastic_runge_kutta),
    ]:
        paths = []
        start = time.time()

        for _ in range(num_trials):
            path = solver(drift_fn, diffusion_fn, x0, ts, dt=dt)
            paths.append(path[-1])  # Final state

        elapsed = time.time() - start
        finals = torch.stack(paths)

        results[name] = {
            "mean": finals.mean(dim=0),
            "std": finals.std(dim=0),
            "time_seconds": elapsed,
            "time_per_trial": elapsed / num_trials,
        }

    return results
