"""
Visualization
=============

Visualization utilities for Neural SDE models:
    - Sample path distributions (fan charts)
    - Learned volatility surfaces
    - Uncertainty bands / prediction intervals
    - Training curves
    - Path statistics comparison
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_sample_paths(
    paths: np.ndarray,
    ts: Optional[np.ndarray] = None,
    title: str = "Neural SDE Sample Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
    num_paths_to_show: int = 50,
    alpha: float = 0.3,
    show_mean: bool = True,
    show_ci: bool = True,
    ci_level: float = 0.95,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot sample paths from a Neural SDE with confidence intervals.

    Args:
        paths: Array of shape (num_paths, num_times) or (num_paths, num_times, dim)
        ts: Time points (num_times,). If None, use integer indices.
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        num_paths_to_show: Number of individual paths to plot
        alpha: Transparency for individual paths
        show_mean: Show the mean path
        show_ci: Show confidence interval
        ci_level: Confidence level (e.g., 0.95 for 95%)
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    if isinstance(paths, torch.Tensor):
        paths = paths.detach().cpu().numpy()

    # Handle multi-dimensional paths (use first dimension)
    if paths.ndim == 3:
        paths = paths[:, :, 0]

    num_paths = paths.shape[0]
    num_times = paths.shape[1]

    if ts is None:
        ts = np.arange(num_times)
    elif isinstance(ts, torch.Tensor):
        ts = ts.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual paths
    indices = np.random.choice(
        num_paths, min(num_paths_to_show, num_paths), replace=False
    )
    for idx in indices:
        ax.plot(ts, paths[idx], alpha=alpha, linewidth=0.5, color="steelblue")

    # Plot mean
    if show_mean:
        mean_path = paths.mean(axis=0)
        ax.plot(ts, mean_path, color="darkred", linewidth=2, label="Mean path")

    # Plot confidence interval
    if show_ci:
        alpha_ci = (1 - ci_level) / 2
        lower = np.quantile(paths, alpha_ci, axis=0)
        upper = np.quantile(paths, 1 - alpha_ci, axis=0)
        ax.fill_between(
            ts,
            lower,
            upper,
            alpha=0.2,
            color="orange",
            label=f"{ci_level*100:.0f}% CI",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_volatility_surface(
    model,
    state_range: Tuple[float, float] = (-2.0, 2.0),
    time_range: Tuple[float, float] = (0.0, 1.0),
    resolution: int = 50,
    state_dim_idx: int = 0,
    title: str = "Learned Diffusion Coefficient g(x, t)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot the learned diffusion (volatility) surface g(x, t).

    Creates a heatmap showing how the diffusion coefficient varies
    with state and time.

    Args:
        model: Neural SDE model with a .g(t, y) or .diffusion_net(y, t) method
        state_range: Range of state values to evaluate
        time_range: Range of time values
        resolution: Grid resolution
        state_dim_idx: Which state dimension to vary (others held at 0)
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    states = np.linspace(state_range[0], state_range[1], resolution)
    times = np.linspace(time_range[0], time_range[1], resolution)

    # Evaluate diffusion on grid
    state_dim = getattr(model, "state_dim", 1)
    vol_surface = np.zeros((resolution, resolution))

    model.eval()
    with torch.no_grad():
        for i, s in enumerate(states):
            for j, t in enumerate(times):
                state = torch.zeros(1, state_dim)
                state[0, state_dim_idx] = s
                t_tensor = torch.tensor(t)

                if hasattr(model, "g"):
                    sigma = model.g(t_tensor, state)
                elif hasattr(model, "diffusion_net"):
                    sigma = model.diffusion_net(state, t_tensor)
                else:
                    raise AttributeError("Model has no diffusion method")

                vol_surface[i, j] = sigma[0, state_dim_idx].item()

    fig, ax = plt.subplots(figsize=figsize)

    T, S = np.meshgrid(times, states)
    c = ax.pcolormesh(T, S, vol_surface, cmap="magma", shading="auto")

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("State (x)")
    ax.set_title(title)
    fig.colorbar(c, ax=ax, label="Diffusion coefficient")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_drift_field(
    model,
    state_range: Tuple[float, float] = (-2.0, 2.0),
    time_range: Tuple[float, float] = (0.0, 1.0),
    resolution: int = 20,
    state_dim_idx: int = 0,
    title: str = "Learned Drift Field f(x, t)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot the learned drift as a vector field.

    Args:
        model: Neural SDE model
        state_range: Range of state values
        time_range: Range of time values
        resolution: Grid resolution
        state_dim_idx: Which state dimension
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    states = np.linspace(state_range[0], state_range[1], resolution)
    times = np.linspace(time_range[0], time_range[1], resolution)

    state_dim = getattr(model, "state_dim", 1)
    T, S = np.meshgrid(times, states)
    U = np.ones_like(T)  # Time always moves forward
    V = np.zeros_like(S)  # Drift in state dimension

    model.eval()
    with torch.no_grad():
        for i, s in enumerate(states):
            for j, t in enumerate(times):
                state = torch.zeros(1, state_dim)
                state[0, state_dim_idx] = s
                t_tensor = torch.tensor(t)

                if hasattr(model, "f"):
                    drift = model.f(t_tensor, state)
                elif hasattr(model, "drift_net"):
                    drift = model.drift_net(state, t_tensor)
                else:
                    raise AttributeError("Model has no drift method")

                V[i, j] = drift[0, state_dim_idx].item()

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize arrows for visibility
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U_norm = U / magnitude.clip(min=1e-8)
    V_norm = V / magnitude.clip(min=1e-8)

    q = ax.quiver(
        T, S, U_norm, V_norm, magnitude, cmap="coolwarm", alpha=0.8, scale=30
    )

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("State (x)")
    ax.set_title(title)
    fig.colorbar(q, ax=ax, label="Drift magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_uncertainty_bands(
    observed: np.ndarray,
    mean_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ts_obs: Optional[np.ndarray] = None,
    ts_pred: Optional[np.ndarray] = None,
    title: str = "Neural SDE Prediction with Uncertainty",
    xlabel: str = "Time",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Plot observations with prediction uncertainty bands.

    Args:
        observed: Observed values
        mean_pred: Mean prediction
        lower: Lower prediction bound
        upper: Upper prediction bound
        ts_obs: Time points for observations
        ts_pred: Time points for predictions
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional save path
        figsize: Figure size
    """
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().cpu().numpy()
    if isinstance(mean_pred, torch.Tensor):
        mean_pred = mean_pred.detach().cpu().numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.detach().cpu().numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.detach().cpu().numpy()

    if ts_obs is None:
        ts_obs = np.arange(len(observed))
    if ts_pred is None:
        ts_pred = np.arange(len(observed), len(observed) + len(mean_pred))

    fig, ax = plt.subplots(figsize=figsize)

    # Observed data
    ax.plot(ts_obs, observed, "k-", linewidth=1.5, label="Observed", zorder=3)

    # Prediction mean
    ax.plot(
        ts_pred,
        mean_pred,
        "r--",
        linewidth=1.5,
        label="Prediction (mean)",
        zorder=2,
    )

    # Uncertainty bands
    ax.fill_between(
        ts_pred,
        lower,
        upper,
        alpha=0.25,
        color="red",
        label="95% prediction interval",
        zorder=1,
    )

    # Dividing line
    ax.axvline(x=ts_obs[-1], color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot training curves (loss, reconstruction, KL, learning rate, beta).

    Args:
        history: Training history dictionary from train_latent_sde
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(len(history.get("train_loss", [])))

    # Total loss
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()

    # Reconstruction loss
    ax = axes[0, 1]
    if "train_recon" in history:
        ax.plot(epochs, history["train_recon"], label="Train recon")
    if "val_recon" in history:
        ax.plot(epochs, history["val_recon"], label="Val recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()

    # KL divergence
    ax = axes[1, 0]
    if "train_kl" in history:
        ax.plot(epochs, history["train_kl"], label="Train KL")
    if "val_kl" in history:
        ax.plot(epochs, history["val_kl"], label="Val KL")
    if "beta" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["beta"], "g--", alpha=0.5, label="beta")
        ax2.set_ylabel("Beta (KL weight)")
        ax2.legend(loc="upper left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence + Beta Annealing")
    ax.legend(loc="upper right")

    # Learning rate
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(epochs, history["lr"])
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_path_statistics(
    real_paths: np.ndarray,
    generated_paths: np.ndarray,
    labels: Tuple[str, str] = ("Real", "Neural SDE"),
    title: str = "Path Statistics Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Compare statistical properties of real vs. generated paths.

    Plots histograms of returns, autocorrelation, and QQ plots.

    Args:
        real_paths: Real data paths (num_paths, num_times) or (num_times,)
        generated_paths: Generated paths (num_paths, num_times)
        labels: Labels for real and generated data
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    if isinstance(real_paths, torch.Tensor):
        real_paths = real_paths.detach().cpu().numpy()
    if isinstance(generated_paths, torch.Tensor):
        generated_paths = generated_paths.detach().cpu().numpy()

    # Compute returns
    if real_paths.ndim == 1:
        real_returns = np.diff(real_paths) / real_paths[:-1]
    else:
        real_returns = np.diff(real_paths, axis=-1) / real_paths[:, :-1].clip(min=1e-10)
        real_returns = real_returns.flatten()

    gen_returns = np.diff(generated_paths, axis=-1) / generated_paths[:, :-1].clip(
        min=1e-10
    )
    gen_returns = gen_returns.flatten()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Return distributions
    ax = axes[0, 0]
    bins = np.linspace(
        min(real_returns.min(), gen_returns.min()),
        max(real_returns.max(), gen_returns.max()),
        100,
    )
    ax.hist(real_returns, bins=bins, alpha=0.5, density=True, label=labels[0])
    ax.hist(gen_returns, bins=bins, alpha=0.5, density=True, label=labels[1])
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution")
    ax.legend()

    # Log-scale return distribution (for tail comparison)
    ax = axes[0, 1]
    ax.hist(
        real_returns,
        bins=bins,
        alpha=0.5,
        density=True,
        label=labels[0],
        log=True,
    )
    ax.hist(
        gen_returns,
        bins=bins,
        alpha=0.5,
        density=True,
        label=labels[1],
        log=True,
    )
    ax.set_xlabel("Return")
    ax.set_ylabel("Log Density")
    ax.set_title("Return Distribution (Log Scale)")
    ax.legend()

    # Autocorrelation of absolute returns
    ax = axes[1, 0]
    max_lag = min(50, len(real_returns) // 2)

    real_abs = np.abs(real_returns - real_returns.mean())
    gen_abs = np.abs(gen_returns - gen_returns.mean())

    real_acf = np.correlate(real_abs, real_abs, mode="full")
    real_acf = real_acf[len(real_acf) // 2 :]
    real_acf = real_acf[:max_lag] / real_acf[0]

    gen_acf = np.correlate(gen_abs, gen_abs, mode="full")
    gen_acf = gen_acf[len(gen_acf) // 2 :]
    gen_acf = gen_acf[:max_lag] / gen_acf[0]

    ax.plot(range(max_lag), real_acf, label=labels[0])
    ax.plot(range(max_lag), gen_acf, label=labels[1])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of |Returns|")
    ax.legend()

    # Summary statistics table
    ax = axes[1, 1]
    ax.axis("off")

    stats_data = []
    for name, returns in [(labels[0], real_returns), (labels[1], gen_returns)]:
        from scipy import stats as sp_stats

        stats_data.append(
            [
                name,
                f"{returns.mean():.6f}",
                f"{returns.std():.6f}",
                f"{sp_stats.skew(returns):.3f}",
                f"{sp_stats.kurtosis(returns):.3f}",
                f"{returns.min():.4f}",
                f"{returns.max():.4f}",
            ]
        )

    table = ax.table(
        cellText=stats_data,
        colLabels=["Data", "Mean", "Std", "Skew", "Kurt", "Min", "Max"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Summary Statistics")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_regime_detection(
    prices: np.ndarray,
    volatilities: np.ndarray,
    ts: Optional[np.ndarray] = None,
    vol_threshold: float = None,
    title: str = "Neural SDE Regime Detection",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot price with Neural SDE-detected volatility regimes.

    Args:
        prices: Price series
        volatilities: Learned volatility from diffusion network
        ts: Time points
        vol_threshold: Threshold for high/low volatility (default: median)
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    if isinstance(prices, torch.Tensor):
        prices = prices.detach().cpu().numpy()
    if isinstance(volatilities, torch.Tensor):
        volatilities = volatilities.detach().cpu().numpy()

    if ts is None:
        ts = np.arange(len(prices))

    if vol_threshold is None:
        vol_threshold = np.median(volatilities)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Price with regime coloring
    high_vol = volatilities > vol_threshold

    for i in range(1, len(ts)):
        color = "red" if high_vol[i] else "green"
        ax1.plot(ts[i - 1 : i + 1], prices[i - 1 : i + 1], color=color, linewidth=1)

    ax1.set_ylabel("Price")
    ax1.set_title(f"{title} - Price with Regime Coloring")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="green", label="Low volatility regime"),
        Line2D([0], [0], color="red", label="High volatility regime"),
    ]
    ax1.legend(handles=legend_elements)

    # Volatility
    ax2.plot(ts, volatilities, color="purple", linewidth=1)
    ax2.axhline(y=vol_threshold, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(ts, 0, volatilities, alpha=0.2, color="purple")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Learned Volatility g(x,t)")
    ax2.set_title("Diffusion Coefficient (Learned Volatility)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Demo with synthetic data
    print("Generating visualization demo...")

    # Create synthetic paths
    np.random.seed(42)
    num_paths = 200
    num_times = 100
    ts = np.linspace(0, 1, num_times)

    # GBM-like paths
    mu, sigma = 0.1, 0.3
    dt = ts[1] - ts[0]
    paths = np.zeros((num_paths, num_times))
    paths[:, 0] = 100

    for t in range(1, num_times):
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * dW
        )

    print("Plotting sample paths...")
    plot_sample_paths(
        paths,
        ts=ts,
        title="Synthetic GBM Sample Paths",
        ylabel="Price",
    )

    print("Demo complete!")
