"""
Data Loader
============

Data loading utilities for stock market and cryptocurrency (Bybit) data.
Supports both historical OHLCV data and real-time price feeds.

Data sources:
    - Bybit V5 API (crypto perpetual futures)
    - Yahoo Finance via yfinance (stocks)
    - Synthetic data for testing
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ═══════════════════════════════════════════════════════════════════════
# BYBIT CRYPTO DATA
# ═══════════════════════════════════════════════════════════════════════


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch candlestick (OHLCV) data from Bybit V5 API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Candle interval. Options:
            '1','3','5','15','30','60','120','240','360','720','D','W','M'
        limit: Number of candles to fetch (max 1000 per request)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        category: Market category ('linear' for USDT perpetuals)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover
    """
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_time is not None:
        params["start"] = start_time
    if end_time is not None:
        params["end"] = end_time

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise Exception(f"Bybit API error: {data['retMsg']}")

    klines = data["result"]["list"]

    if not klines:
        return pd.DataFrame()

    # Parse klines: [timestamp, open, high, low, close, volume, turnover]
    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    # Sort by timestamp ascending (Bybit returns newest first)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def fetch_bybit_extended(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    days: int = 90,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch extended historical data from Bybit by paginating through the API.

    Args:
        symbol: Trading pair
        interval: Candle interval
        days: Number of days of history to fetch
        category: Market category

    Returns:
        DataFrame with full history
    """
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    current_end = end_time

    while current_end > start_time:
        try:
            df = fetch_bybit_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                end_time=current_end,
                category=category,
            )

            if df.empty:
                break

            all_data.append(df)

            # Move end time to before the earliest fetched candle
            earliest_ts = int(df["timestamp"].iloc[0].timestamp() * 1000)
            if earliest_ts >= current_end:
                break
            current_end = earliest_ts - 1

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    result = result.reset_index(drop=True)

    return result


def fetch_bybit_orderbook(
    symbol: str = "BTCUSDT",
    category: str = "linear",
    depth: int = 50,
) -> Dict:
    """
    Fetch current order book from Bybit.

    Args:
        symbol: Trading pair
        category: Market category
        depth: Order book depth (1, 25, 50, 100, 200)

    Returns:
        Dictionary with 'bids' and 'asks' as lists of [price, quantity]
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": category,
        "symbol": symbol,
        "limit": depth,
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise Exception(f"Bybit API error: {data['retMsg']}")

    result = data["result"]
    return {
        "bids": [[float(p), float(q)] for p, q in result["b"]],
        "asks": [[float(p), float(q)] for p, q in result["a"]],
        "timestamp": int(result["ts"]),
    }


def fetch_bybit_funding_rate(
    symbol: str = "BTCUSDT",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Bybit.

    Funding rate is unique to perpetual futures and reflects the premium/discount
    of the perp price vs. spot price. It is predictive of short-term price moves.

    Args:
        symbol: Trading pair
        limit: Number of records

    Returns:
        DataFrame with funding rate history
    """
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit,
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise Exception(f"Bybit API error: {data['retMsg']}")

    records = data["result"]["list"]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["fundingRateTimestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(int), unit="ms"
    )
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.sort_values("fundingRateTimestamp").reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════
# STOCK MARKET DATA
# ═══════════════════════════════════════════════════════════════════════


def fetch_stock_data(
    symbol: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch stock market data using yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'AAPL', 'TSLA')
        period: Data period ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max')
        interval: Data interval ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Standardize column names
        rename_map = {}
        if "date" in df.columns:
            rename_map["date"] = "timestamp"
        if "datetime" in df.columns:
            rename_map["datetime"] = "timestamp"

        df = df.rename(columns=rename_map)

        # Keep only essential columns
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

        return df

    except ImportError:
        print("yfinance not installed. Using synthetic stock data.")
        return generate_synthetic_data(symbol=symbol, num_points=500)


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING FOR NEURAL SDE
# ═══════════════════════════════════════════════════════════════════════


def prepare_sde_data(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    window_size: int = 60,
    prediction_horizon: int = 24,
    normalize: bool = True,
    log_transform_price: bool = True,
) -> Dict:
    """
    Prepare data for Neural SDE training.

    Converts OHLCV data into overlapping windows suitable for
    training a Latent Neural SDE.

    Args:
        df: DataFrame with OHLCV data
        feature_columns: Columns to use as features (default: derived features)
        window_size: Length of each training sequence
        prediction_horizon: Steps to predict into the future
        normalize: Whether to normalize features
        log_transform_price: Apply log transform to price

    Returns:
        Dictionary with:
            - observations: (num_windows, window_size, num_features)
            - targets: (num_windows, prediction_horizon, num_features)
            - ts: (window_size,) normalized time points
            - scaler_params: normalization parameters
    """
    import torch

    # Compute derived features
    df = df.copy()

    if log_transform_price:
        df["log_price"] = np.log(df["close"])
        df["log_return"] = df["log_price"].diff()
    else:
        df["log_return"] = df["close"].pct_change()

    # Realized volatility (rolling std of returns)
    df["realized_vol"] = df["log_return"].rolling(window=20).std()

    # Volume ratio (relative to 20-period moving average)
    vol_ma = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / vol_ma.clip(lower=1e-10)

    # Price momentum (return over 10 periods)
    df["momentum"] = df["close"].pct_change(periods=10)

    # High-low range (intraday volatility proxy)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    # Select features
    if feature_columns is None:
        feature_columns = [
            "log_return",
            "realized_vol",
            "volume_ratio",
            "momentum",
            "hl_range",
        ]

    features = df[feature_columns].values

    # Normalize
    scaler_params = {}
    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0).clip(min=1e-8)
        features = (features - mean) / std
        scaler_params = {"mean": mean, "std": std}

    # Create overlapping windows
    total_len = window_size + prediction_horizon
    num_windows = len(features) - total_len + 1

    if num_windows <= 0:
        raise ValueError(
            f"Not enough data: {len(features)} rows, need {total_len}. "
            f"Reduce window_size or prediction_horizon."
        )

    observations = np.zeros((num_windows, window_size, len(feature_columns)))
    targets = np.zeros((num_windows, prediction_horizon, len(feature_columns)))

    for i in range(num_windows):
        observations[i] = features[i : i + window_size]
        targets[i] = features[
            i + window_size : i + window_size + prediction_horizon
        ]

    # Normalized time points
    ts = np.linspace(0, 1, window_size)

    return {
        "observations": torch.tensor(observations, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "ts": torch.tensor(ts, dtype=torch.float32),
        "scaler_params": scaler_params,
        "feature_columns": feature_columns,
        "raw_df": df,
    }


def create_dataloaders(
    data_dict: Dict,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    """
    Create PyTorch DataLoaders from prepared data.

    Args:
        data_dict: Output from prepare_sde_data
        batch_size: Batch size
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Dictionary with train_loader, val_loader, test_loader, ts
    """
    from torch.utils.data import DataLoader, TensorDataset

    observations = data_dict["observations"]
    targets = data_dict["targets"]
    ts = data_dict["ts"]

    n = len(observations)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Time-series split (no shuffling to respect temporal order)
    train_obs = observations[:n_train]
    train_targets = targets[:n_train]

    val_obs = observations[n_train : n_train + n_val]
    val_targets = targets[n_train : n_train + n_val]

    test_obs = observations[n_train + n_val :]
    test_targets = targets[n_train + n_val :]

    train_loader = DataLoader(
        TensorDataset(train_obs, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_obs, val_targets),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(test_obs, test_targets),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Data split: train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "ts": ts,
    }


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════


def generate_synthetic_data(
    symbol: str = "SYNTH",
    num_points: int = 1000,
    dt: float = 1.0 / 252,
    mu: float = 0.1,
    sigma: float = 0.2,
    s0: float = 100.0,
    regime_switching: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic financial time series for testing.

    Optionally includes regime-switching dynamics that Neural SDEs
    should be able to capture.

    Args:
        symbol: Name for the synthetic asset
        num_points: Number of data points
        dt: Time increment (1/252 for daily)
        mu: Base drift parameter
        sigma: Base volatility parameter
        s0: Initial price
        regime_switching: Include regime changes for more realistic data

    Returns:
        DataFrame with OHLCV-like data
    """
    np.random.seed(42)

    prices = np.zeros(num_points)
    prices[0] = s0

    if regime_switching:
        # Two regimes: bull (high drift, low vol) and bear (low drift, high vol)
        regime = 0  # Start in bull regime
        for i in range(1, num_points):
            # Regime transition probabilities
            if regime == 0:  # Bull
                drift = mu
                vol = sigma * 0.7
                if np.random.random() < 0.02:  # 2% chance to switch to bear
                    regime = 1
            else:  # Bear
                drift = -mu * 0.5
                vol = sigma * 1.5
                if np.random.random() < 0.03:  # 3% chance to switch to bull
                    regime = 0

            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i - 1] * np.exp(
                (drift - 0.5 * vol ** 2) * dt + vol * dW
            )
    else:
        # Simple GBM
        dW = np.random.normal(0, np.sqrt(dt), num_points - 1)
        for i in range(1, num_points):
            prices[i] = prices[i - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * dW[i - 1]
            )

    # Generate OHLCV-like data from close prices
    noise_scale = sigma * np.sqrt(dt) * 0.3
    highs = prices * (1 + np.abs(np.random.normal(0, noise_scale, num_points)))
    lows = prices * (1 - np.abs(np.random.normal(0, noise_scale, num_points)))
    opens = np.roll(prices, 1)
    opens[0] = s0
    volumes = np.random.lognormal(mean=15, sigma=1, size=num_points)

    timestamps = pd.date_range(
        start="2022-01-01", periods=num_points, freq="1h"
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )

    return df


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def load_crypto_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    days: int = 90,
) -> Dict:
    """
    Load and prepare crypto data for Neural SDE training in one call.

    Args:
        symbol: Bybit trading pair
        interval: Candle interval
        days: Days of history

    Returns:
        Prepared data dictionary ready for training
    """
    print(f"Fetching {symbol} data from Bybit ({days} days, {interval}m interval)...")

    try:
        df = fetch_bybit_extended(symbol=symbol, interval=interval, days=days)
        if df.empty:
            raise ValueError("No data returned from Bybit")
        print(f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    except Exception as e:
        print(f"Failed to fetch from Bybit: {e}")
        print("Falling back to synthetic data...")
        df = generate_synthetic_data(symbol=symbol, num_points=days * 24)

    return prepare_sde_data(df)


def load_stock_data(
    symbol: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> Dict:
    """
    Load and prepare stock data for Neural SDE training in one call.

    Args:
        symbol: Ticker symbol
        period: Data period
        interval: Data interval

    Returns:
        Prepared data dictionary ready for training
    """
    print(f"Fetching {symbol} stock data ({period}, {interval} interval)...")

    df = fetch_stock_data(symbol=symbol, period=period, interval=interval)
    if df.empty:
        print("Falling back to synthetic data...")
        df = generate_synthetic_data(symbol=symbol, num_points=500)

    return prepare_sde_data(df)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Neural SDE Data Loader - Example Usage")
    print("=" * 60)

    # Try Bybit data
    print("\n--- Bybit BTCUSDT (1h candles, last 30 days) ---")
    try:
        df = fetch_bybit_klines(symbol="BTCUSDT", interval="60", limit=100)
        print(f"Fetched {len(df)} candles")
        print(df.head())
        print(f"\nPrice range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    except Exception as e:
        print(f"Bybit fetch failed: {e}")

    # Synthetic data fallback
    print("\n--- Synthetic Data ---")
    df_synth = generate_synthetic_data(num_points=500, regime_switching=True)
    print(f"Generated {len(df_synth)} data points")
    print(df_synth.head())

    # Prepare for training
    print("\n--- Preparing for Neural SDE training ---")
    data = prepare_sde_data(df_synth, window_size=60, prediction_horizon=24)
    print(f"Observations shape: {data['observations'].shape}")
    print(f"Targets shape: {data['targets'].shape}")
    print(f"Time points: {data['ts'].shape}")
    print(f"Features: {data['feature_columns']}")
