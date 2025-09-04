//! # Neural SDE Trading
//!
//! Neural Stochastic Differential Equations for cryptocurrency trading.
//!
//! This crate implements:
//! - Neural SDE model with learnable drift and diffusion networks
//! - SDE solvers (Euler-Maruyama, Milstein, SRK)
//! - Bybit API client for market data
//! - Backtesting engine
//!
//! ## Mathematical Formulation
//!
//! ```text
//! dX(t) = f_θ(X(t), t) dt + g_φ(X(t), t) dW(t)
//! ```
//!
//! where f_θ is the drift network and g_φ is the diffusion network,
//! both parameterized by neural network weights.

pub mod api;
pub mod backtest;
pub mod model;
pub mod solvers;

// Re-exports
pub use api::BybitClient;
pub use backtest::{BacktestConfig, BacktestEngine, BacktestResult};
pub use model::{DiffusionNetwork, DriftNetwork, NeuralSDE};
pub use solvers::{euler_maruyama, milstein, stochastic_runge_kutta, SolverMethod};

/// Activation functions used in the neural networks.
pub mod activations {
    /// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
    #[inline]
    pub fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// Softplus activation: ln(1 + exp(x))
    /// Ensures positive output for diffusion coefficients.
    #[inline]
    pub fn softplus(x: f64) -> f64 {
        if x > 20.0 {
            x // Avoid overflow
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// ReLU activation: max(0, x)
    #[inline]
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Tanh activation
    #[inline]
    pub fn tanh_act(x: f64) -> f64 {
        x.tanh()
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Utility functions for data processing.
pub mod utils {
    use ndarray::Array1;

    /// Compute log returns from a price series.
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Compute simple returns from a price series.
    pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Compute rolling standard deviation (realized volatility).
    pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![0.0; data.len()];
        }

        let mut result = vec![0.0; window - 1];

        for i in (window - 1)..data.len() {
            let slice = &data[(i + 1 - window)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (slice.len() - 1) as f64;
            result.push(variance.sqrt());
        }

        result
    }

    /// Normalize data to zero mean and unit variance.
    pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
        let n = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n;
        let std: f64 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
            .sqrt()
            .max(1e-8);

        let normalized: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();
        (normalized, mean, std)
    }

    /// Compute running maximum (for drawdown calculation).
    pub fn running_max(data: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut current_max = f64::NEG_INFINITY;

        for &val in data {
            current_max = current_max.max(val);
            result.push(current_max);
        }

        result
    }

    /// Compute maximum drawdown.
    pub fn max_drawdown(equity: &[f64]) -> f64 {
        let peaks = running_max(equity);
        let mut max_dd = 0.0f64;

        for (eq, peak) in equity.iter().zip(peaks.iter()) {
            let dd = (eq - peak) / peak;
            max_dd = max_dd.min(dd);
        }

        max_dd
    }

    /// Compute the Sharpe ratio.
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let std: f64 = {
            let var: f64 = returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / (returns.len() - 1).max(1) as f64;
            var.sqrt()
        };

        if std < 1e-10 {
            return 0.0;
        }

        let annual_mean = mean * periods_per_year;
        let annual_std = std * periods_per_year.sqrt();

        (annual_mean - risk_free_rate) / annual_std
    }
}

/// API module for Bybit exchange.
pub mod api {
    use anyhow::Result;
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};

    /// OHLCV candlestick data.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Kline {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub turnover: f64,
    }

    /// Bybit API response wrapper.
    #[derive(Debug, Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        #[serde(rename = "retMsg")]
        ret_msg: String,
        result: serde_json::Value,
    }

    /// Bybit API client.
    pub struct BybitClient {
        base_url: String,
        client: reqwest::blocking::Client,
    }

    impl BybitClient {
        /// Create a new Bybit client.
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
                client: reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .expect("Failed to build HTTP client"),
            }
        }

        /// Fetch candlestick data.
        ///
        /// # Arguments
        /// * `symbol` - Trading pair (e.g., "BTCUSDT")
        /// * `interval` - Candle interval (e.g., "60" for 1 hour)
        /// * `limit` - Number of candles (max 1000)
        /// * `start` - Optional start timestamp (ms)
        /// * `end` - Optional end timestamp (ms)
        pub fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: u32,
            start: Option<i64>,
            end: Option<i64>,
        ) -> Result<Vec<Kline>> {
            let mut params = vec![
                ("category", "linear".to_string()),
                ("symbol", symbol.to_string()),
                ("interval", interval.to_string()),
                ("limit", limit.to_string()),
            ];

            if let Some(s) = start {
                params.push(("start", s.to_string()));
            }
            if let Some(e) = end {
                params.push(("end", e.to_string()));
            }

            let url = format!("{}/v5/market/kline", self.base_url);
            let response: BybitResponse = self
                .client
                .get(&url)
                .query(&params)
                .send()?
                .json()?;

            if response.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", response.ret_msg);
            }

            let list = response.result["list"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("No kline data in response"))?;

            let mut klines: Vec<Kline> = list
                .iter()
                .filter_map(|item| {
                    let arr = item.as_array()?;
                    if arr.len() < 7 {
                        return None;
                    }
                    Some(Kline {
                        timestamp: arr[0].as_str()?.parse().ok()?,
                        open: arr[1].as_str()?.parse().ok()?,
                        high: arr[2].as_str()?.parse().ok()?,
                        low: arr[3].as_str()?.parse().ok()?,
                        close: arr[4].as_str()?.parse().ok()?,
                        volume: arr[5].as_str()?.parse().ok()?,
                        turnover: arr[6].as_str()?.parse().ok()?,
                    })
                })
                .collect();

            // Sort by timestamp ascending (Bybit returns newest first)
            klines.sort_by_key(|k| k.timestamp);

            Ok(klines)
        }

        /// Fetch extended historical data by paginating through the API.
        pub fn fetch_extended(
            &self,
            symbol: &str,
            interval: &str,
            days: u32,
        ) -> Result<Vec<Kline>> {
            let now_ms = Utc::now().timestamp_millis();
            let start_ms = now_ms - (days as i64) * 24 * 60 * 60 * 1000;

            let mut all_klines = Vec::new();
            let mut current_end = now_ms;

            while current_end > start_ms {
                let klines = self.fetch_klines(symbol, interval, 1000, None, Some(current_end))?;

                if klines.is_empty() {
                    break;
                }

                let earliest = klines[0].timestamp;
                all_klines.extend(klines);

                if earliest >= current_end {
                    break;
                }
                current_end = earliest - 1;

                // Rate limiting
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            // Sort and deduplicate
            all_klines.sort_by_key(|k| k.timestamp);
            all_klines.dedup_by_key(|k| k.timestamp);

            Ok(all_klines)
        }

        /// Save klines to CSV file.
        pub fn save_to_csv(&self, klines: &[Kline], path: &str) -> Result<()> {
            let mut writer = csv::Writer::from_path(path)?;

            writer.write_record(&[
                "timestamp", "open", "high", "low", "close", "volume", "turnover",
            ])?;

            for k in klines {
                writer.write_record(&[
                    k.timestamp.to_string(),
                    k.open.to_string(),
                    k.high.to_string(),
                    k.low.to_string(),
                    k.close.to_string(),
                    k.volume.to_string(),
                    k.turnover.to_string(),
                ])?;
            }

            writer.flush()?;
            Ok(())
        }

        /// Load klines from CSV file.
        pub fn load_from_csv(path: &str) -> Result<Vec<Kline>> {
            let mut reader = csv::Reader::from_path(path)?;
            let mut klines = Vec::new();

            for result in reader.records() {
                let record = result?;
                klines.push(Kline {
                    timestamp: record[0].parse()?,
                    open: record[1].parse()?,
                    high: record[2].parse()?,
                    low: record[3].parse()?,
                    close: record[4].parse()?,
                    volume: record[5].parse()?,
                    turnover: record[6].parse()?,
                });
            }

            Ok(klines)
        }
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Neural network model for SDEs.
pub mod model {
    use ndarray::{Array1, Array2};
    use rand::Rng;
    use rand_distr::StandardNormal;

    use crate::activations::{silu, softplus};

    /// Dense (fully connected) neural network layer.
    #[derive(Debug, Clone)]
    pub struct DenseLayer {
        pub weights: Array2<f64>,
        pub bias: Array1<f64>,
    }

    impl DenseLayer {
        /// Create a new layer with Xavier initialization.
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let scale = (2.0 / (input_dim + output_dim) as f64).sqrt() * 0.1;

            let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
                rng.sample::<f64, _>(StandardNormal) * scale
            });

            let bias = Array1::zeros(output_dim);

            Self { weights, bias }
        }

        /// Forward pass: output = weights * input + bias
        pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
            self.weights.dot(input) + &self.bias
        }
    }

    /// Drift network f_θ(x, t) -> R^d.
    ///
    /// Maps (state, time) to the deterministic drift component.
    #[derive(Debug, Clone)]
    pub struct DriftNetwork {
        layers: Vec<DenseLayer>,
        state_dim: usize,
    }

    impl DriftNetwork {
        /// Create a new drift network.
        ///
        /// # Arguments
        /// * `state_dim` - Dimension of the state vector
        /// * `hidden_dim` - Hidden layer size
        /// * `num_layers` - Number of layers (including output)
        pub fn new(state_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
            let mut layers = Vec::new();
            let input_dim = state_dim + 1; // +1 for time

            for i in 0..num_layers {
                let in_dim = if i == 0 { input_dim } else { hidden_dim };
                let out_dim = if i == num_layers - 1 {
                    state_dim
                } else {
                    hidden_dim
                };
                layers.push(DenseLayer::new(in_dim, out_dim));
            }

            Self { layers, state_dim }
        }

        /// Compute the drift f_θ(x, t).
        pub fn forward(&self, state: &Array1<f64>, t: f64) -> Array1<f64> {
            // Concatenate state and time
            let mut input = Array1::zeros(state.len() + 1);
            input.slice_mut(ndarray::s![..state.len()])
                .assign(state);
            input[state.len()] = t;

            let mut x = input;
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    x.mapv_inplace(silu);
                }
            }

            x
        }
    }

    /// Diffusion network g_φ(x, t) -> R^d₊.
    ///
    /// Maps (state, time) to the stochastic diffusion coefficient.
    /// Uses Softplus activation to ensure positivity.
    #[derive(Debug, Clone)]
    pub struct DiffusionNetwork {
        layers: Vec<DenseLayer>,
        state_dim: usize,
        min_diffusion: f64,
    }

    impl DiffusionNetwork {
        /// Create a new diffusion network.
        pub fn new(
            state_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
            min_diffusion: f64,
        ) -> Self {
            let mut layers = Vec::new();
            let input_dim = state_dim + 1;

            for i in 0..num_layers {
                let in_dim = if i == 0 { input_dim } else { hidden_dim };
                let out_dim = if i == num_layers - 1 {
                    state_dim
                } else {
                    hidden_dim
                };
                layers.push(DenseLayer::new(in_dim, out_dim));
            }

            Self {
                layers,
                state_dim,
                min_diffusion,
            }
        }

        /// Compute the diffusion coefficient g_φ(x, t).
        pub fn forward(&self, state: &Array1<f64>, t: f64) -> Array1<f64> {
            let mut input = Array1::zeros(state.len() + 1);
            input.slice_mut(ndarray::s![..state.len()])
                .assign(state);
            input[state.len()] = t;

            let mut x = input;
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    x.mapv_inplace(silu);
                } else {
                    // Final layer: apply Softplus + minimum diffusion
                    x.mapv_inplace(|v| softplus(v) + self.min_diffusion);
                }
            }

            x
        }
    }

    /// Neural SDE model combining drift and diffusion networks.
    ///
    /// dX(t) = f_θ(X(t), t) dt + g_φ(X(t), t) dW(t)
    #[derive(Debug, Clone)]
    pub struct NeuralSDE {
        pub drift: DriftNetwork,
        pub diffusion: DiffusionNetwork,
        pub state_dim: usize,
    }

    impl NeuralSDE {
        /// Create a new Neural SDE model.
        pub fn new(state_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
            Self {
                drift: DriftNetwork::new(state_dim, hidden_dim, num_layers),
                diffusion: DiffusionNetwork::new(state_dim, hidden_dim, num_layers, 1e-3),
                state_dim,
            }
        }

        /// Compute drift at (state, t).
        pub fn f(&self, state: &Array1<f64>, t: f64) -> Array1<f64> {
            self.drift.forward(state, t)
        }

        /// Compute diffusion at (state, t).
        pub fn g(&self, state: &Array1<f64>, t: f64) -> Array1<f64> {
            self.diffusion.forward(state, t)
        }

        /// Generate a single sample path.
        pub fn sample_path(
            &self,
            x0: &Array1<f64>,
            ts: &[f64],
            dt: f64,
        ) -> Vec<Array1<f64>> {
            crate::solvers::euler_maruyama(
                |x, t| self.f(x, t),
                |x, t| self.g(x, t),
                x0,
                ts,
                dt,
            )
        }

        /// Generate multiple sample paths.
        pub fn sample_paths(
            &self,
            x0: &Array1<f64>,
            ts: &[f64],
            dt: f64,
            num_paths: usize,
        ) -> Vec<Vec<Array1<f64>>> {
            (0..num_paths)
                .map(|_| self.sample_path(x0, ts, dt))
                .collect()
        }

        /// Generate paths in parallel using rayon.
        pub fn sample_paths_parallel(
            &self,
            x0: &Array1<f64>,
            ts: &[f64],
            dt: f64,
            num_paths: usize,
        ) -> Vec<Vec<Array1<f64>>> {
            use rayon::prelude::*;

            (0..num_paths)
                .into_par_iter()
                .map(|_| {
                    crate::solvers::euler_maruyama(
                        |x, t| self.f(x, t),
                        |x, t| self.g(x, t),
                        x0,
                        ts,
                        dt,
                    )
                })
                .collect()
        }

        /// Predict future distribution (mean, lower, upper bounds).
        pub fn predict_distribution(
            &self,
            x0: &Array1<f64>,
            ts: &[f64],
            dt: f64,
            num_samples: usize,
            confidence: f64,
        ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>, Vec<Array1<f64>>) {
            let paths = self.sample_paths_parallel(x0, ts, dt, num_samples);
            let num_times = ts.len();
            let dim = self.state_dim;

            let mut means = Vec::with_capacity(num_times);
            let mut lowers = Vec::with_capacity(num_times);
            let mut uppers = Vec::with_capacity(num_times);

            let alpha = (1.0 - confidence) / 2.0;

            for t_idx in 0..num_times {
                let mut mean = Array1::zeros(dim);
                let mut values_per_dim: Vec<Vec<f64>> = (0..dim).map(|_| Vec::new()).collect();

                for path in &paths {
                    let state = &path[t_idx];
                    mean = mean + state;
                    for d in 0..dim {
                        values_per_dim[d].push(state[d]);
                    }
                }

                mean /= num_samples as f64;

                let mut lower = Array1::zeros(dim);
                let mut upper = Array1::zeros(dim);

                for d in 0..dim {
                    values_per_dim[d].sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n = values_per_dim[d].len();
                    let lo_idx = ((alpha * n as f64) as usize).min(n - 1);
                    let hi_idx = (((1.0 - alpha) * n as f64) as usize).min(n - 1);
                    lower[d] = values_per_dim[d][lo_idx];
                    upper[d] = values_per_dim[d][hi_idx];
                }

                means.push(mean);
                lowers.push(lower);
                uppers.push(upper);
            }

            (means, lowers, uppers)
        }
    }
}

/// SDE numerical solvers.
pub mod solvers {
    use ndarray::Array1;
    use rand::Rng;
    use rand_distr::StandardNormal;

    /// Solver method enumeration.
    #[derive(Debug, Clone, Copy)]
    pub enum SolverMethod {
        EulerMaruyama,
        Milstein,
        StochasticRungeKutta,
    }

    /// Euler-Maruyama SDE solver.
    ///
    /// X(t+dt) = X(t) + f(X,t)*dt + g(X,t)*dW
    ///
    /// Strong order: 0.5, Weak order: 1.0
    pub fn euler_maruyama<F, G>(
        drift: F,
        diffusion: G,
        x0: &Array1<f64>,
        ts: &[f64],
        dt: f64,
    ) -> Vec<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
        G: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut rng = rand::thread_rng();
        let mut trajectory = vec![x0.clone()];
        let mut x = x0.clone();
        let mut t_current = ts[0];
        let mut time_idx = 1;

        while time_idx < ts.len() {
            let t_target = ts[time_idx];

            while t_current < t_target - 1e-10 {
                let actual_dt = dt.min(t_target - t_current);

                let f = drift(&x, t_current);
                let g = diffusion(&x, t_current);

                // Brownian increment
                let dw: Array1<f64> = Array1::from_shape_fn(x.len(), |_| {
                    rng.sample::<f64, _>(StandardNormal) * actual_dt.sqrt()
                });

                // Euler-Maruyama step
                x = x + &f * actual_dt + &g * &dw;

                t_current += actual_dt;
            }

            trajectory.push(x.clone());
            time_idx += 1;
        }

        trajectory
    }

    /// Milstein SDE solver.
    ///
    /// X(t+dt) = X(t) + f*dt + g*dW + 0.5*g*g'*(dW^2 - dt)
    ///
    /// Strong order: 1.0 (requires diffusion derivative)
    pub fn milstein<F, G>(
        drift: F,
        diffusion: G,
        x0: &Array1<f64>,
        ts: &[f64],
        dt: f64,
    ) -> Vec<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
        G: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let eps = 1e-5;
        let mut rng = rand::thread_rng();
        let mut trajectory = vec![x0.clone()];
        let mut x = x0.clone();
        let mut t_current = ts[0];
        let mut time_idx = 1;

        while time_idx < ts.len() {
            let t_target = ts[time_idx];

            while t_current < t_target - 1e-10 {
                let actual_dt = dt.min(t_target - t_current);

                let f = drift(&x, t_current);
                let g = diffusion(&x, t_current);

                // Approximate dg/dx via finite differences
                let x_plus = &x + eps;
                let g_plus = diffusion(&x_plus, t_current);
                let g_prime = (&g_plus - &g) / eps;

                // Brownian increment
                let dw: Array1<f64> = Array1::from_shape_fn(x.len(), |_| {
                    rng.sample::<f64, _>(StandardNormal) * actual_dt.sqrt()
                });

                // Milstein step
                let dw_sq = &dw * &dw;
                x = x + &f * actual_dt + &g * &dw + &g * &g_prime * (dw_sq - actual_dt) * 0.5;

                t_current += actual_dt;
            }

            trajectory.push(x.clone());
            time_idx += 1;
        }

        trajectory
    }

    /// Stochastic Runge-Kutta solver.
    ///
    /// Two-stage method achieving strong order 1.0 without derivatives.
    pub fn stochastic_runge_kutta<F, G>(
        drift: F,
        diffusion: G,
        x0: &Array1<f64>,
        ts: &[f64],
        dt: f64,
    ) -> Vec<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
        G: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut rng = rand::thread_rng();
        let mut trajectory = vec![x0.clone()];
        let mut x = x0.clone();
        let mut t_current = ts[0];
        let mut time_idx = 1;

        while time_idx < ts.len() {
            let t_target = ts[time_idx];

            while t_current < t_target - 1e-10 {
                let actual_dt = dt.min(t_target - t_current);

                // Brownian increment
                let dw: Array1<f64> = Array1::from_shape_fn(x.len(), |_| {
                    rng.sample::<f64, _>(StandardNormal) * actual_dt.sqrt()
                });

                // Stage 1
                let f1 = drift(&x, t_current);
                let g1 = diffusion(&x, t_current);
                let k1 = &f1 * actual_dt + &g1 * &dw;

                // Supporting value
                let x_hat = &x + &k1;

                // Stage 2
                let f2 = drift(&x_hat, t_current + actual_dt);
                let g2 = diffusion(&x_hat, t_current + actual_dt);
                let k2 = &f2 * actual_dt + &g2 * &dw;

                // SRK update
                x = x + (&k1 + &k2) * 0.5;

                t_current += actual_dt;
            }

            trajectory.push(x.clone());
            time_idx += 1;
        }

        trajectory
    }

    /// Geometric Brownian Motion (GBM) exact simulation for benchmarking.
    pub fn geometric_brownian_motion(
        mu: f64,
        sigma: f64,
        s0: f64,
        ts: &[f64],
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut path = vec![s0];
        let mut s = s0;

        for i in 1..ts.len() {
            let dt = ts[i] - ts[i - 1];
            let dw: f64 = rng.sample::<f64, _>(StandardNormal) * dt.sqrt();
            s *= ((mu - 0.5 * sigma * sigma) * dt + sigma * dw).exp();
            path.push(s);
        }

        path
    }

    /// Unified solver interface.
    pub fn solve<F, G>(
        drift: F,
        diffusion: G,
        x0: &Array1<f64>,
        ts: &[f64],
        dt: f64,
        method: SolverMethod,
    ) -> Vec<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
        G: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        match method {
            SolverMethod::EulerMaruyama => euler_maruyama(drift, diffusion, x0, ts, dt),
            SolverMethod::Milstein => milstein(drift, diffusion, x0, ts, dt),
            SolverMethod::StochasticRungeKutta => {
                stochastic_runge_kutta(drift, diffusion, x0, ts, dt)
            }
        }
    }
}

/// Backtesting engine.
pub mod backtest {
    use crate::api::Kline;
    use crate::model::NeuralSDE;
    use crate::utils;
    use ndarray::Array1;

    /// Backtest configuration.
    #[derive(Debug, Clone)]
    pub struct BacktestConfig {
        pub prediction_horizon: usize,
        pub num_monte_carlo_paths: usize,
        pub rebalance_frequency: usize,
        pub signal_threshold_long: f64,
        pub signal_threshold_short: f64,
        pub max_position: f64,
        pub max_volatility: f64,
        pub transaction_cost_bps: f64,
        pub initial_capital: f64,
        pub sde_dt: f64,
    }

    impl Default for BacktestConfig {
        fn default() -> Self {
            Self {
                prediction_horizon: 24,
                num_monte_carlo_paths: 200,
                rebalance_frequency: 4,
                signal_threshold_long: 0.5,
                signal_threshold_short: -0.5,
                max_position: 1.0,
                max_volatility: 0.05,
                transaction_cost_bps: 10.0,
                initial_capital: 100_000.0,
                sde_dt: 0.02,
            }
        }
    }

    /// Single trade record.
    #[derive(Debug, Clone)]
    pub struct Trade {
        pub entry_time: usize,
        pub exit_time: usize,
        pub entry_price: f64,
        pub exit_price: f64,
        pub position_size: f64,
        pub pnl: f64,
        pub reason: String,
    }

    /// Backtest results.
    #[derive(Debug)]
    pub struct BacktestResult {
        pub equity_curve: Vec<f64>,
        pub positions: Vec<f64>,
        pub signals: Vec<f64>,
        pub trades: Vec<Trade>,
        pub total_return: f64,
        pub annual_return: f64,
        pub annual_volatility: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub num_trades: usize,
    }

    impl BacktestResult {
        /// Print a summary of the backtest results.
        pub fn print_summary(&self) {
            println!("\n{}", "=".repeat(60));
            println!("BACKTEST RESULTS");
            println!("{}", "=".repeat(60));
            println!("Total Return:       {:>10.2}%", self.total_return * 100.0);
            println!("Annual Return:      {:>10.2}%", self.annual_return * 100.0);
            println!(
                "Annual Volatility:  {:>10.2}%",
                self.annual_volatility * 100.0
            );
            println!("Sharpe Ratio:       {:>10.3}", self.sharpe_ratio);
            println!("Max Drawdown:       {:>10.2}%", self.max_drawdown * 100.0);
            println!("{}", "-".repeat(60));
            println!("Number of Trades:   {:>10}", self.num_trades);
            println!("Win Rate:           {:>10.1}%", self.win_rate * 100.0);
            println!("{}", "=".repeat(60));
        }
    }

    /// Backtesting engine for Neural SDE strategies.
    pub struct BacktestEngine {
        model: NeuralSDE,
        config: BacktestConfig,
    }

    impl BacktestEngine {
        pub fn new(model: NeuralSDE, config: BacktestConfig) -> Self {
            Self { model, config }
        }

        /// Run the backtest on historical price data.
        pub fn run(&self, prices: &[f64], features: &[Array1<f64>]) -> BacktestResult {
            let n = prices.len();
            let cfg = &self.config;

            let mut equity = vec![0.0; n];
            equity[0] = cfg.initial_capital;

            let mut positions = vec![0.0; n];
            let mut signals = vec![0.0; n];
            let mut trades = Vec::new();

            let mut current_position = 0.0;
            let mut entry_price = 0.0;

            // Time points for SDE prediction
            let horizon_ts: Vec<f64> = (0..=cfg.prediction_horizon)
                .map(|i| i as f64 / cfg.prediction_horizon as f64)
                .collect();

            for t in 1..n {
                // Update equity
                let price_ret = (prices[t] - prices[t - 1]) / prices[t - 1];
                let pnl = current_position * equity[t - 1] * price_ret;
                equity[t] = equity[t - 1] + pnl;

                // Rebalance
                if t % cfg.rebalance_frequency == 0
                    && t + cfg.prediction_horizon < n
                    && t < features.len()
                {
                    // Generate Monte Carlo paths
                    let state = &features[t];
                    let paths = self.model.sample_paths_parallel(
                        state,
                        &horizon_ts,
                        cfg.sde_dt,
                        cfg.num_monte_carlo_paths,
                    );

                    // Compute signal from path distribution
                    let final_values: Vec<f64> = paths
                        .iter()
                        .map(|p| p.last().unwrap()[0])
                        .collect();

                    let mean: f64 =
                        final_values.iter().sum::<f64>() / final_values.len() as f64;
                    let exp_ret = mean - state[0];

                    let variance: f64 = final_values
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>()
                        / (final_values.len() - 1) as f64;
                    let pred_vol = variance.sqrt();

                    let signal = if pred_vol > 1e-8 {
                        exp_ret / pred_vol
                    } else {
                        0.0
                    };

                    signals[t] = signal;

                    // Position sizing
                    let mut target = if signal > cfg.signal_threshold_long {
                        (signal / 3.0).min(cfg.max_position)
                    } else if signal < cfg.signal_threshold_short {
                        (signal / 3.0).max(-cfg.max_position)
                    } else {
                        0.0
                    };

                    // Volatility scaling
                    if pred_vol > cfg.max_volatility {
                        target *= cfg.max_volatility / pred_vol;
                    }

                    // Execute trade
                    if (target - current_position).abs() > 0.01 {
                        let cost = (target - current_position).abs()
                            * equity[t]
                            * cfg.transaction_cost_bps
                            / 10000.0;
                        equity[t] -= cost;

                        if current_position != 0.0 {
                            let trade_ret = (prices[t] - entry_price) / entry_price;
                            trades.push(Trade {
                                entry_time: 0,
                                exit_time: t,
                                entry_price,
                                exit_price: prices[t],
                                position_size: current_position,
                                pnl: current_position * equity[t - 1] * trade_ret,
                                reason: "rebalance".to_string(),
                            });
                        }

                        current_position = target;
                        entry_price = if target != 0.0 { prices[t] } else { 0.0 };
                    }
                }

                positions[t] = current_position;
            }

            // Compute metrics
            let returns = utils::simple_returns(&equity);
            let total_return = equity.last().unwrap() / equity[0] - 1.0;
            let periods_per_year = 252.0 * 24.0;
            let n_periods = returns.len() as f64;

            let annual_return =
                (1.0 + total_return).powf(periods_per_year / n_periods) - 1.0;

            let annual_volatility = {
                let mean: f64 = returns.iter().sum::<f64>() / n_periods;
                let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / (n_periods - 1.0);
                var.sqrt() * periods_per_year.sqrt()
            };

            let sharpe = if annual_volatility > 0.0 {
                annual_return / annual_volatility
            } else {
                0.0
            };

            let max_dd = utils::max_drawdown(&equity);

            let winning = trades.iter().filter(|t| t.pnl > 0.0).count();
            let win_rate = if trades.is_empty() {
                0.0
            } else {
                winning as f64 / trades.len() as f64
            };

            BacktestResult {
                equity_curve: equity,
                positions,
                signals,
                trades: trades.clone(),
                total_return,
                annual_return,
                annual_volatility,
                sharpe_ratio: sharpe,
                max_drawdown: max_dd,
                win_rate,
                num_trades: trades.len(),
            }
        }
    }
}
