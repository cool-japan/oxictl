use crate::core::scalar::ControlScalar;

/// Resolver-to-Digital Converter (RDC) using a software tracking loop.
///
/// A resolver outputs two signals:
///   sin_sig = R · sin(θ)
///   cos_sig = R · cos(θ)
///
/// The tracking loop (type-2 PLL) drives the angle estimate θ̂ to match θ:
///   ε = cos(θ̂)·sin_sig − sin(θ̂)·cos_sig  ≈  R·sin(θ − θ̂)  ≈  R·(θ − θ̂)
///
/// The loop integrates ε → ω̂ → θ̂ (velocity and angle estimate).
#[derive(Debug, Clone, Copy)]
pub struct ResolverDecoder<S: ControlScalar> {
    /// Proportional gain (rad/s per unit error).
    pub kp: S,
    /// Integral gain (rad/s² per unit-second error).
    pub ki: S,
    /// Estimated electrical angle (rad).
    theta: S,
    /// Estimated angular velocity (rad/s).
    omega: S,
    /// Integral state for type-2 loop.
    integrator: S,
}

impl<S: ControlScalar> ResolverDecoder<S> {
    /// Create a new resolver decoder.
    ///
    /// Typical gains: kp ≈ 2000, ki ≈ kp²/4 for critically damped response.
    pub fn new(kp: S, ki: S) -> Self {
        Self {
            kp,
            ki,
            theta: S::ZERO,
            omega: S::ZERO,
            integrator: S::ZERO,
        }
    }

    /// Update the tracking loop.
    ///
    /// - `sin_sig`: resolver sin output (R·sin θ, where R is amplitude)
    /// - `cos_sig`: resolver cos output (R·cos θ)
    /// - `dt`: time step (s)
    pub fn update(&mut self, sin_sig: S, cos_sig: S, dt: S) {
        // Tracking error: ε = cos(θ̂)·sin − sin(θ̂)·cos = R·sin(θ − θ̂)
        let sin_hat = self.theta.sin();
        let cos_hat = self.theta.cos();
        let error = cos_hat * sin_sig - sin_hat * cos_sig;

        // PI filter → velocity estimate
        self.integrator += error * dt;
        let omega_corr = self.kp * error + self.ki * self.integrator;
        self.omega = omega_corr;

        // Integrate velocity → angle
        self.theta += self.omega * dt;
        // Wrap angle to [-π, π]
        while self.theta > S::PI {
            self.theta -= S::TWO * S::PI;
        }
        while self.theta < -S::PI {
            self.theta += S::TWO * S::PI;
        }
    }

    /// Estimated electrical angle (rad), range [−π, π].
    pub fn theta(&self) -> S {
        self.theta
    }

    /// Estimated angular velocity (rad/s).
    pub fn omega(&self) -> S {
        self.omega
    }

    pub fn reset(&mut self) {
        self.theta = S::ZERO;
        self.omega = S::ZERO;
        self.integrator = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_constant_angle() {
        // Simulate resolver at fixed angle 1.0 rad
        let target = 1.0_f64;
        let amp = 1.0_f64;
        let sin_sig = amp * target.sin();
        let cos_sig = amp * target.cos();

        let mut dec = ResolverDecoder::new(200.0_f64, 10_000.0);
        let dt = 1e-4_f64;
        for _ in 0..50_000 {
            dec.update(sin_sig, cos_sig, dt);
        }
        assert!(
            (dec.theta() - target).abs() < 0.01,
            "theta={:.4} (expected {:.4})",
            dec.theta(),
            target
        );
    }

    #[test]
    fn tracks_rotating_resolver() {
        // Simulate resolver rotating at ω = 100 rad/s
        let omega_true = 100.0_f64;
        let amp = 1.0_f64;
        let mut dec = ResolverDecoder::new(2000.0_f64, 100_000.0);
        let dt = 1e-5_f64;

        let mut theta_true = 0.0_f64;
        for _ in 0..100_000 {
            theta_true += omega_true * dt;
            let sin_sig = amp * theta_true.sin();
            let cos_sig = amp * theta_true.cos();
            dec.update(sin_sig, cos_sig, dt);
        }

        // After settling, velocity estimate should be close to true
        assert!(
            (dec.omega() - omega_true).abs() < 10.0,
            "omega_est={:.2} (true={:.2})",
            dec.omega(),
            omega_true
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut dec = ResolverDecoder::new(200.0_f64, 10_000.0);
        dec.update(0.5, 0.866, 1e-4);
        dec.reset();
        assert_eq!(dec.theta(), 0.0);
        assert_eq!(dec.omega(), 0.0);
    }
}
