//! Noise models for sensor simulation: Gaussian, Uniform, Outlier.
//!
//! Uses an XOR-shift PRNG (deterministic, no_std) and Box-Muller transform
//! for Gaussian sampling. Runs under the std feature.
#![cfg(feature = "std")]

/// XOR-shift PRNG (period 2^64 - 1).
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create with non-zero seed (if seed == 0, uses 1).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for mantissa precision
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Gaussian noise generator using the Box-Muller transform.
///
/// Produces samples from N(0, sigma^2) on each call to `sample`.
pub struct GaussianNoise {
    rng: Xorshift64,
    spare: Option<f64>,
}

impl GaussianNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            spare: None,
        }
    }

    /// Sample from N(0, sigma^2) using Box-Muller.
    pub fn sample(&mut self, sigma: f64) -> f64 {
        // Use spare sample if available
        if let Some(spare) = self.spare.take() {
            return spare * sigma;
        }

        // Box-Muller: generate two independent normal samples
        let u1 = loop {
            let u = self.rng.next_f64();
            if u > 1e-15 {
                break u;
            }
        };
        let u2 = self.rng.next_f64();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        let z0 = mag * theta.cos();
        let z1 = mag * theta.sin();
        self.spare = Some(z1);
        z0 * sigma
    }

    /// Add Gaussian noise N(0, amplitude^2) to signal.
    pub fn add_noise(&mut self, signal: f64, amplitude: f64) -> f64 {
        signal + self.sample(amplitude)
    }
}

/// Uniform noise generator producing samples in [-amplitude, +amplitude].
pub struct UniformNoise {
    rng: Xorshift64,
}

impl UniformNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
        }
    }

    /// Sample uniformly from [-amplitude, +amplitude].
    pub fn sample(&mut self, amplitude: f64) -> f64 {
        // Map [0,1) to [-1,1)
        let u = self.rng.next_f64() * 2.0 - 1.0;
        u * amplitude
    }

    /// Add uniform noise to signal.
    pub fn add_noise(&mut self, signal: f64, amplitude: f64) -> f64 {
        signal + self.sample(amplitude)
    }
}

/// Outlier (spike) noise: injects an outlier with a configurable probability.
///
/// When an outlier is injected the returned value is ±amplitude (sign randomised).
pub struct OutlierNoise {
    rng: Xorshift64,
    /// Probability of outlier per sample (0.0 – 1.0).
    pub rate: f64,
    /// Outlier amplitude.
    pub amplitude: f64,
}

impl OutlierNoise {
    pub fn new(seed: u64, rate: f64, amplitude: f64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            rate,
            amplitude,
        }
    }

    /// Returns signal unchanged or replaces it with ±amplitude outlier.
    pub fn add_noise(&mut self, signal: f64) -> f64 {
        let u = self.rng.next_f64();
        if u < self.rate {
            // Inject outlier; sign from another random draw
            let sign = if self.rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
            sign * self.amplitude
        } else {
            signal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_gaussian_mean_and_variance() {
        let mut noise = GaussianNoise::new(12345);
        let sigma = 2.0_f64;
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| noise.sample(sigma)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05 * sigma, "mean={mean}");
        assert!(
            (var - sigma * sigma).abs() < 0.15 * sigma * sigma,
            "var={var}"
        );
    }

    #[test]
    fn test_uniform_bounds() {
        let mut noise = UniformNoise::new(99);
        let amp = 3.0_f64;
        for _ in 0..10_000 {
            let s = noise.sample(amp);
            assert!(s >= -amp && s < amp, "out of bounds: {s}");
        }
    }

    #[test]
    fn test_outlier_rate() {
        let rate = 0.05_f64;
        let mut noise = OutlierNoise::new(7, rate, 100.0);
        let signal = 1.0_f64;
        let n = 50_000;
        let outliers = (0..n)
            .filter(|_| {
                let v = noise.add_noise(signal);
                (v - signal).abs() > 50.0
            })
            .count();
        let measured_rate = outliers as f64 / n as f64;
        assert!(
            (measured_rate - rate).abs() < 0.01,
            "measured_rate={measured_rate}"
        );
    }

    #[test]
    fn test_uniform_noise_add() {
        let mut noise = UniformNoise::new(1);
        let signal = 5.0_f64;
        let amp = 0.1_f64;
        let result = noise.add_noise(signal, amp);
        assert!((result - signal).abs() <= amp + 1e-12);
    }
}
