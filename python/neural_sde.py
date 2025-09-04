"""
Neural SDE Model
================

Core Neural SDE implementation with learnable drift and diffusion networks.
Uses the torchsde library for SDE solving and adjoint-based training.

Mathematical formulation:
    dX(t) = f_theta(X(t), t) dt + g_phi(X(t), t) dW(t)

where:
    f_theta: R^d x R -> R^d        (drift network)
    g_phi:   R^d x R -> R^(d x m)  (diffusion network)
    W(t):    m-dimensional Brownian motion
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DriftNetwork(nn.Module):
    """
    Drift network f_theta(x, t) -> R^d.

    Maps (state, time) to the drift vector. This represents the
    deterministic component of the SDE dynamics.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim

        layers = []
        in_dim = state_dim + 1  # +1 for time

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else state_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialize weights with small values for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift.

        Args:
            x: State tensor of shape (batch, state_dim)
            t: Time scalar or tensor of shape (batch, 1)

        Returns:
            Drift vector of shape (batch, state_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class DiffusionNetwork(nn.Module):
    """
    Diffusion network g_phi(x, t) -> R^d_+.

    Maps (state, time) to the diffusion coefficient. Uses Softplus
    activation to ensure positivity (required for valid SDE dynamics).

    Supports diagonal and scalar diffusion modes:
    - diagonal: independent noise per state dimension (default)
    - scalar: single noise coefficient applied to all dimensions
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        diffusion_type: str = "diagonal",
        min_diffusion: float = 1e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.diffusion_type = diffusion_type
        self.min_diffusion = min_diffusion

        output_dim = state_dim if diffusion_type == "diagonal" else 1

        layers = []
        in_dim = state_dim + 1

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient.

        Args:
            x: State tensor of shape (batch, state_dim)
            t: Time scalar or tensor of shape (batch, 1)

        Returns:
            Diffusion coefficient of shape (batch, state_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        inp = torch.cat([x, t], dim=-1)
        raw = self.net(inp)
        sigma = self.softplus(raw) + self.min_diffusion

        if self.diffusion_type == "scalar":
            sigma = sigma.expand(-1, self.state_dim)

        return sigma


class NeuralSDE(nn.Module):
    """
    Neural Stochastic Differential Equation.

    dX(t) = f_theta(X(t), t) dt + g_phi(X(t), t) dW(t)

    Compatible with torchsde.sdeint for solving and training
    via the stochastic adjoint method.

    Args:
        state_dim: Dimension of the state vector
        hidden_dim: Hidden layer size for drift and diffusion networks
        num_layers: Number of layers in each network
        diffusion_type: 'diagonal' or 'scalar'
        noise_type: 'diagonal' or 'general' (for torchsde compatibility)
        sde_type: 'ito' or 'stratonovich'
    """

    # Required by torchsde
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        diffusion_type: str = "diagonal",
        sde_type_param: str = "ito",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.sde_type = sde_type_param

        self.drift_net = DriftNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.diffusion_net = DiffusionNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            diffusion_type=diffusion_type,
        )

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient for torchsde interface.

        Args:
            t: Current time (scalar tensor)
            y: Current state (batch, state_dim)

        Returns:
            Drift vector (batch, state_dim)
        """
        return self.drift_net(y, t)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient for torchsde interface.

        Args:
            t: Current time (scalar tensor)
            y: Current state (batch, state_dim)

        Returns:
            Diffusion coefficient (batch, state_dim) for diagonal noise
        """
        return self.diffusion_net(y, t)

    def forward(
        self,
        y0: torch.Tensor,
        ts: torch.Tensor,
        dt: float = 0.01,
        method: str = "euler",
        adaptive: bool = False,
    ) -> torch.Tensor:
        """
        Solve the Neural SDE forward in time.

        Args:
            y0: Initial state (batch, state_dim)
            ts: Time points to evaluate at (num_times,)
            dt: Step size for the solver
            method: Solver method ('euler', 'milstein', 'srk')
            adaptive: Whether to use adaptive step size

        Returns:
            Solution trajectory (num_times, batch, state_dim)
        """
        try:
            import torchsde

            return torchsde.sdeint(
                self,
                y0,
                ts,
                dt=dt,
                method=method,
                adaptive=adaptive,
            )
        except ImportError:
            # Fallback to manual Euler-Maruyama if torchsde is not installed
            return self._euler_maruyama(y0, ts, dt)

    def _euler_maruyama(
        self, y0: torch.Tensor, ts: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Fallback Euler-Maruyama solver when torchsde is not available.

        Args:
            y0: Initial state (batch, state_dim)
            ts: Time points (num_times,)
            dt: Step size

        Returns:
            Trajectory (num_times, batch, state_dim)
        """
        trajectory = [y0]
        y = y0
        t_current = ts[0]
        time_idx = 1

        while time_idx < len(ts):
            t_next = min(t_current + dt, ts[time_idx])
            actual_dt = (t_next - t_current).item()

            if actual_dt > 0:
                drift = self.f(t_current, y)
                diffusion = self.g(t_current, y)
                dW = torch.randn_like(y) * math.sqrt(actual_dt)
                y = y + drift * actual_dt + diffusion * dW

            t_current = t_next

            if torch.isclose(t_current, ts[time_idx]):
                trajectory.append(y.clone())
                time_idx += 1

        return torch.stack(trajectory)

    def sample_paths(
        self,
        y0: torch.Tensor,
        ts: torch.Tensor,
        num_paths: int = 100,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """
        Generate multiple sample paths from the Neural SDE.

        Args:
            y0: Initial state (state_dim,) or (1, state_dim)
            ts: Time points (num_times,)
            num_paths: Number of paths to generate
            dt: Solver step size

        Returns:
            Paths tensor (num_paths, num_times, state_dim)
        """
        if y0.dim() == 1:
            y0 = y0.unsqueeze(0)

        # Expand y0 for all paths
        y0_batch = y0.expand(num_paths, -1)

        # Solve SDE
        trajectories = self.forward(y0_batch, ts, dt=dt)

        # trajectories shape: (num_times, num_paths, state_dim)
        # Transpose to (num_paths, num_times, state_dim)
        return trajectories.permute(1, 0, 2)

    def predict_distribution(
        self,
        y0: torch.Tensor,
        ts: torch.Tensor,
        num_samples: int = 500,
        dt: float = 0.01,
        confidence: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the distribution of future states.

        Args:
            y0: Current state (state_dim,)
            ts: Future time points
            num_samples: Number of Monte Carlo samples
            dt: Solver step size
            confidence: Confidence level for prediction intervals

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
            Each of shape (num_times, state_dim)
        """
        with torch.no_grad():
            paths = self.sample_paths(y0, ts, num_paths=num_samples, dt=dt)

        # paths: (num_samples, num_times, state_dim)
        mean = paths.mean(dim=0)

        alpha = (1 - confidence) / 2
        lower = torch.quantile(paths, alpha, dim=0)
        upper = torch.quantile(paths, 1 - alpha, dim=0)

        return mean, lower, upper


class FinancialNeuralSDE(NeuralSDE):
    """
    Neural SDE specialized for financial time series.

    Extends NeuralSDE with:
    - Log-price dynamics (ensures positivity)
    - Separate volatility output
    - Feature-augmented state (volume, indicators)
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__(
            state_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            diffusion_type="diagonal",
        )
        self.num_features = num_features

        # Additional network for extracting trading signals
        self.signal_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # [expected_return, volatility, signal]
            nn.Tanh(),
        )

    def extract_signal(self, state: torch.Tensor) -> dict:
        """
        Extract trading signals from the current state.

        Args:
            state: Current state (batch, num_features)

        Returns:
            Dictionary with expected_return, volatility, and signal
        """
        output = self.signal_net(state)
        return {
            "expected_return": output[:, 0],
            "volatility": output[:, 1].abs(),
            "signal": output[:, 2],
        }
