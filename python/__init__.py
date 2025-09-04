"""
Neural SDE Trading - Chapter 147
================================

Neural Stochastic Differential Equations for financial time series modeling
and trading strategy development.

Modules:
    neural_sde   - Core Neural SDE model (drift + diffusion networks)
    latent_sde   - Latent SDE for time series with encoder-decoder
    train        - Training loop with stochastic adjoint method
    data_loader  - Stock and Bybit crypto data loading
    sde_solvers  - Euler-Maruyama, Milstein SDE solvers
    visualize    - Path distributions, volatility surfaces, uncertainty bands
    backtest     - Trading strategy backtesting
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"
