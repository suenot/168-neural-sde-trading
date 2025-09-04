"""
Backtest
========

Trading strategy backtesting using Neural SDE predictions.

Strategy logic:
    1. At each rebalance time, generate Monte Carlo paths from Neural SDE
    2. Compute expected return and predicted volatility from path distribution
    3. Size positions using Sharpe-like signal: signal = E[r] / sigma
    4. Apply risk management: reduce positions when volatility is high
    5. Track PnL, drawdown, and other performance metrics

Supports:
    - Long/short trading
    - Transaction costs
    - Position limits
    - Multiple assets
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from neural_sde import FinancialNeuralSDE, NeuralSDE


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""

    # Strategy parameters
    prediction_horizon: int = 24  # Steps to predict ahead
    num_monte_carlo_paths: int = 200  # MC paths for prediction
    rebalance_frequency: int = 4  # Rebalance every N steps
    signal_threshold_long: float = 0.5  # Minimum signal to go long
    signal_threshold_short: float = -0.5  # Minimum signal to go short
    max_position: float = 1.0  # Maximum position size (fraction of capital)
    min_position: float = -1.0  # Minimum position size (for shorts)

    # Risk management
    max_volatility: float = 0.05  # Max vol threshold (reduce position above)
    stop_loss_pct: float = 0.05  # Stop loss as fraction of entry price
    take_profit_pct: float = 0.10  # Take profit as fraction of entry price

    # Costs
    transaction_cost_bps: float = 10.0  # Transaction cost in basis points
    slippage_bps: float = 5.0  # Slippage in basis points

    # SDE solver
    sde_dt: float = 0.02  # SDE solver step size

    # Capital
    initial_capital: float = 100_000.0


@dataclass
class Trade:
    """Record of a single trade."""

    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    reason: str  # 'signal', 'stop_loss', 'take_profit', 'rebalance'


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    # Equity curve
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    signals: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_vols: np.ndarray = field(default_factory=lambda: np.array([]))

    # Trades
    trades: List[Trade] = field(default_factory=list)

    # Summary metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0

    def compute_metrics(self, periods_per_year: float = 252 * 24):
        """Compute all summary metrics from the equity curve."""
        if len(self.equity_curve) < 2:
            return

        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        self.returns = returns

        # Returns
        self.total_return = (
            self.equity_curve[-1] / self.equity_curve[0] - 1
        )
        n_periods = len(returns)
        self.annual_return = (1 + self.total_return) ** (
            periods_per_year / n_periods
        ) - 1

        # Volatility
        self.annual_volatility = returns.std() * np.sqrt(periods_per_year)

        # Sharpe
        if self.annual_volatility > 0:
            self.sharpe_ratio = self.annual_return / self.annual_volatility
        else:
            self.sharpe_ratio = 0.0

        # Drawdown
        cummax = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - cummax) / cummax
        self.max_drawdown = drawdown.min()

        # Calmar
        if self.max_drawdown < 0:
            self.calmar_ratio = self.annual_return / abs(self.max_drawdown)

        # Trade statistics
        if self.trades:
            self.num_trades = len(self.trades)
            pnls = [t.pnl for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            self.win_rate = len(wins) / len(pnls) if pnls else 0
            self.avg_trade_pnl = np.mean(pnls) if pnls else 0

            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1e-10
            self.profit_factor = total_wins / total_losses

    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"Total Return:       {self.total_return*100:>10.2f}%\n"
            f"Annual Return:      {self.annual_return*100:>10.2f}%\n"
            f"Annual Volatility:  {self.annual_volatility*100:>10.2f}%\n"
            f"Sharpe Ratio:       {self.sharpe_ratio:>10.3f}\n"
            f"Max Drawdown:       {self.max_drawdown*100:>10.2f}%\n"
            f"Calmar Ratio:       {self.calmar_ratio:>10.3f}\n"
            f"{'─'*60}\n"
            f"Number of Trades:   {self.num_trades:>10d}\n"
            f"Win Rate:           {self.win_rate*100:>10.1f}%\n"
            f"Profit Factor:      {self.profit_factor:>10.3f}\n"
            f"Avg Trade PnL:      ${self.avg_trade_pnl:>10.2f}\n"
            f"{'='*60}\n"
        )


class NeuralSDEBacktester:
    """
    Backtesting engine for Neural SDE trading strategies.

    The strategy generates Monte Carlo paths from the Neural SDE,
    computes expected return and volatility, and sizes positions
    based on a signal = E[r] / sigma.
    """

    def __init__(
        self,
        model: NeuralSDE,
        config: Optional[BacktestConfig] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config or BacktestConfig()
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def _generate_signal(
        self, current_state: np.ndarray, ts_forward: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Generate trading signal from Monte Carlo SDE paths.

        Args:
            current_state: Current market state features
            ts_forward: Forward time points for prediction

        Returns:
            Tuple of (signal, expected_return, predicted_volatility)
        """
        state_tensor = torch.tensor(
            current_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Generate Monte Carlo paths
        paths = self.model.sample_paths(
            state_tensor,
            ts_forward,
            num_paths=self.config.num_monte_carlo_paths,
            dt=self.config.sde_dt,
        )

        # paths: (num_paths, num_times, state_dim)
        # Use the first state dimension (typically log-return or price)
        final_states = paths[:, -1, 0].cpu().numpy()
        initial_state = current_state[0]

        # Expected return over the horizon
        expected_return = float(np.mean(final_states) - initial_state)

        # Predicted volatility
        predicted_vol = float(np.std(final_states))

        # Signal: risk-adjusted expected return
        if predicted_vol > 1e-8:
            signal = expected_return / predicted_vol
        else:
            signal = 0.0

        return signal, expected_return, predicted_vol

    def run(
        self,
        data: pd.DataFrame,
        features: np.ndarray,
        prices: np.ndarray,
        ts: Optional[torch.Tensor] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data: DataFrame with market data
            features: Prepared feature array (num_times, num_features)
            prices: Close price array (num_times,)
            ts: Time points for SDE (if None, auto-generated)

        Returns:
            BacktestResult with all metrics and curves
        """
        n = len(prices)
        cfg = self.config

        if ts is None:
            horizon_ts = torch.linspace(0, 1, cfg.prediction_horizon + 1)
        else:
            # Use normalized time points for the prediction horizon
            horizon_ts = torch.linspace(
                0, 1, cfg.prediction_horizon + 1, device=self.device
            )

        # Initialize tracking arrays
        equity = np.zeros(n)
        equity[0] = cfg.initial_capital
        positions_arr = np.zeros(n)
        signals_arr = np.zeros(n)
        vols_arr = np.zeros(n)

        current_position = 0.0
        entry_price = 0.0
        trades = []

        print(f"\nRunning backtest on {n} time steps...")

        for t in range(1, n):
            # Update equity based on position and price change
            if current_position != 0:
                price_return = (prices[t] - prices[t - 1]) / prices[t - 1]
                pnl = current_position * equity[t - 1] * price_return
                equity[t] = equity[t - 1] + pnl
            else:
                equity[t] = equity[t - 1]

            # Check stop loss / take profit
            if current_position != 0 and entry_price > 0:
                price_change = (prices[t] - entry_price) / entry_price
                effective_change = price_change * np.sign(current_position)

                if effective_change < -cfg.stop_loss_pct:
                    # Stop loss hit
                    trade_pnl = current_position * equity[t - 1] * price_change
                    trades.append(
                        Trade(
                            entry_time=0,
                            exit_time=t,
                            entry_price=entry_price,
                            exit_price=prices[t],
                            position_size=current_position,
                            pnl=trade_pnl,
                            pnl_pct=price_change * np.sign(current_position),
                            reason="stop_loss",
                        )
                    )
                    current_position = 0.0
                    entry_price = 0.0

                elif effective_change > cfg.take_profit_pct:
                    # Take profit hit
                    trade_pnl = current_position * equity[t - 1] * price_change
                    trades.append(
                        Trade(
                            entry_time=0,
                            exit_time=t,
                            entry_price=entry_price,
                            exit_price=prices[t],
                            position_size=current_position,
                            pnl=trade_pnl,
                            pnl_pct=price_change * np.sign(current_position),
                            reason="take_profit",
                        )
                    )
                    current_position = 0.0
                    entry_price = 0.0

            # Rebalance at specified frequency
            if t % cfg.rebalance_frequency == 0 and t + cfg.prediction_horizon < n:
                signal, exp_ret, pred_vol = self._generate_signal(
                    features[t], horizon_ts
                )

                signals_arr[t] = signal
                vols_arr[t] = pred_vol

                # Determine target position
                target_position = 0.0

                if signal > cfg.signal_threshold_long:
                    target_position = min(
                        signal / 3.0, cfg.max_position
                    )  # Scale signal to position
                elif signal < cfg.signal_threshold_short:
                    target_position = max(signal / 3.0, cfg.min_position)

                # Reduce position if predicted volatility is too high
                if pred_vol > cfg.max_volatility:
                    vol_scale = cfg.max_volatility / pred_vol
                    target_position *= vol_scale

                # Execute trade if position changes
                if abs(target_position - current_position) > 0.01:
                    # Transaction costs
                    trade_size = abs(target_position - current_position)
                    cost = (
                        trade_size
                        * equity[t]
                        * (cfg.transaction_cost_bps + cfg.slippage_bps)
                        / 10000
                    )
                    equity[t] -= cost

                    if current_position != 0 and entry_price > 0:
                        price_change = (prices[t] - entry_price) / entry_price
                        trade_pnl = (
                            current_position * equity[t - 1] * price_change
                        )
                        trades.append(
                            Trade(
                                entry_time=0,
                                exit_time=t,
                                entry_price=entry_price,
                                exit_price=prices[t],
                                position_size=current_position,
                                pnl=trade_pnl,
                                pnl_pct=price_change
                                * np.sign(current_position),
                                reason="rebalance",
                            )
                        )

                    current_position = target_position
                    entry_price = prices[t] if target_position != 0 else 0.0

            positions_arr[t] = current_position

        # Create result
        result = BacktestResult(
            equity_curve=equity,
            positions=positions_arr,
            signals=signals_arr,
            predicted_vols=vols_arr,
            trades=trades,
        )

        result.compute_metrics()
        return result


def run_comparison_backtest(
    model: NeuralSDE,
    features: np.ndarray,
    prices: np.ndarray,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, BacktestResult]:
    """
    Run comparative backtest: Neural SDE strategy vs. Buy & Hold.

    Args:
        model: Trained Neural SDE model
        features: Feature array
        prices: Price array
        config: Backtest configuration

    Returns:
        Dictionary with 'neural_sde' and 'buy_hold' results
    """
    config = config or BacktestConfig()

    # Neural SDE strategy
    backtester = NeuralSDEBacktester(model, config)
    sde_result = backtester.run(
        data=None,
        features=features,
        prices=prices,
    )

    # Buy & Hold benchmark
    bh_equity = np.zeros(len(prices))
    bh_equity[0] = config.initial_capital
    for t in range(1, len(prices)):
        ret = (prices[t] - prices[t - 1]) / prices[t - 1]
        bh_equity[t] = bh_equity[t - 1] * (1 + ret)

    bh_result = BacktestResult(equity_curve=bh_equity)
    bh_result.compute_metrics()

    return {"neural_sde": sde_result, "buy_hold": bh_result}


def plot_backtest_results(
    results: Dict[str, BacktestResult],
    prices: Optional[np.ndarray] = None,
    title: str = "Neural SDE Trading Backtest",
    save_path: Optional[str] = None,
):
    """
    Plot backtest results with equity curves and metrics.

    Args:
        results: Dictionary of strategy name -> BacktestResult
        prices: Optional price series to overlay
        title: Plot title
        save_path: Optional save path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    colors = ["steelblue", "orange", "green", "red"]

    # Equity curves
    ax = axes[0]
    for i, (name, result) in enumerate(results.items()):
        if len(result.equity_curve) > 0:
            ax.plot(result.equity_curve, label=name, color=colors[i % len(colors)])
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"{title} - Equity Curves")
    ax.legend()

    # Drawdown
    ax = axes[1]
    for i, (name, result) in enumerate(results.items()):
        if len(result.equity_curve) > 0:
            cummax = np.maximum.accumulate(result.equity_curve)
            dd = (result.equity_curve - cummax) / cummax
            ax.fill_between(
                range(len(dd)),
                dd,
                0,
                alpha=0.3,
                label=name,
                color=colors[i % len(colors)],
            )
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()

    # Positions (for the first strategy)
    ax = axes[2]
    first_name = list(results.keys())[0]
    result = results[first_name]
    if len(result.positions) > 0:
        ax.fill_between(
            range(len(result.positions)),
            result.positions,
            0,
            alpha=0.3,
            color="steelblue",
        )
        ax.plot(result.positions, linewidth=0.5, color="steelblue")
    ax.set_ylabel("Position")
    ax.set_xlabel("Time Step")
    ax.set_title(f"{first_name} - Position Size")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    # Print summary for each strategy
    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(result.summary())


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, prepare_sde_data

    print("=" * 60)
    print("Neural SDE Backtest - Demo with Synthetic Data")
    print("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_data(num_points=2000, regime_switching=True)
    data = prepare_sde_data(df, window_size=60, prediction_horizon=24)

    # Create a simple model (untrained, for demo purposes)
    obs_dim = data["observations"].shape[-1]
    model = NeuralSDE(state_dim=obs_dim, hidden_dim=64, num_layers=3)

    # Extract features and prices for backtest
    features = data["observations"][0].numpy()  # Use first window's features
    prices = df["close"].values[60:]  # Prices after the first window

    # Limit to manageable size for demo
    n = min(500, len(prices), len(features))
    features_subset = np.random.randn(n, obs_dim).astype(np.float32)  # Dummy features
    prices_subset = prices[:n]

    # Run backtest
    config = BacktestConfig(
        prediction_horizon=12,
        num_monte_carlo_paths=50,
        rebalance_frequency=8,
        initial_capital=100_000,
    )

    results = run_comparison_backtest(
        model=model,
        features=features_subset,
        prices=prices_subset,
        config=config,
    )

    # Print results
    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(result.summary())
