//! Space vector overmodulation (Mode I and Mode II).
//!
//! Standard SVM operates linearly up to a modulation index of 1.0
//! (voltage amplitude = Vdc/√3). Overmodulation extends beyond this
//! limit toward the six-step waveform at MI ≈ 1.2732 (4/π).
//!
//! Mode I: clamp the reference vector to the inscribed circle of the hexagon.
//! Mode II: redirect the angle toward the nearest vertex for MI > threshold.
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Maximum fundamental output with standard SVM (inscribed circle of hexagon):
/// Vmax = (1/√3) · Vdc
///
/// Note: Vdc/2 is the DC rail to neutral, and SVM gives Vdc/√3 as the
/// max phase-to-neutral amplitude.
pub fn max_linear_voltage<S: ControlScalar>(v_dc: S) -> S {
    // 1/√3 ≈ 0.5773502692
    let inv_sqrt3 = S::from_f64(0.5773502692);
    v_dc * inv_sqrt3
}

/// Maximum voltage with full six-step operation: (4/π) · (Vdc/2) = (2/π) · Vdc.
///
/// This represents the fundamental component of the six-step square wave.
pub fn max_six_step_voltage<S: ControlScalar>(v_dc: S) -> S {
    // 2/π
    let two_over_pi = S::from_f64(core::f64::consts::FRAC_2_PI);
    v_dc * two_over_pi
}

/// Mode I overmodulation: clamp the reference vector amplitude to the
/// inscribed circle (max linear voltage). The angle is preserved.
///
/// # Arguments
/// * `alpha`, `beta` - Reference vector in stationary αβ frame (V).
/// * `v_dc` - DC bus voltage (V).
///
/// # Returns
/// `(alpha_out, beta_out)` clamped to max_linear_voltage radius.
pub fn overmodulate_mode1<S: ControlScalar>(alpha: S, beta: S, v_dc: S) -> (S, S) {
    let v_max = max_linear_voltage(v_dc);
    let mag = (alpha * alpha + beta * beta).sqrt();
    if mag <= v_max || mag <= S::EPSILON {
        (alpha, beta)
    } else {
        let scale = v_max / mag;
        (alpha * scale, beta * scale)
    }
}

/// Mode II overmodulation: phase-angle mapping toward hexagon vertices.
///
/// For modulation index > 1 (Mode II region), the reference angle is
/// progressively pulled toward the nearest 60°-sector midpoint (vertex
/// direction). As MI approaches 4/π, the angle is clamped to the vertex.
///
/// The algorithm:
/// 1. Compute the reference angle θ.
/// 2. Find the sector and its midpoint angle θ_v.
/// 3. Blend: θ_out = θ + blend_factor·(θ_v − θ).
/// 4. Set amplitude to the hexagon boundary at θ_out.
///
/// # Arguments
/// * `mi` - Modulation index (1.0 to 4/π ≈ 1.2732). Values outside this
///   range are clamped.
pub fn overmodulate_mode2<S: ControlScalar>(alpha: S, beta: S, v_dc: S, mi: S) -> (S, S) {
    let v_half = v_dc * S::HALF;

    // Maximum MI: 4/π
    let mi_max = S::from_f64(4.0 / core::f64::consts::PI);
    let mi_clamped = mi.clamp_val(S::ONE, mi_max);

    // Blend factor 0..1: 0 at MI=1, 1 at MI=4/π
    let blend = (mi_clamped - S::ONE) / (mi_max - S::ONE);

    // Reference angle
    let theta = beta.atan2(alpha);

    // Sector index (0..5) and vertex angle (sector midpoint at multiples of 60°)
    let pi_over_3 = S::PI / S::from_f64(3.0);
    let two_pi = S::TWO * S::PI;

    // Normalize theta to [0, 2π)
    let theta_pos = if theta < S::ZERO {
        theta + two_pi
    } else {
        theta
    };

    let sector_f = theta_pos / pi_over_3;
    let sector = sector_f.to_f64() as usize % 6;
    // Vertex angle: sector * 60° + 30°
    let theta_v = S::from_f64((sector as f64 + S::HALF.to_f64()) * 60.0_f64.to_radians());

    // Angle error (sector-relative, wrap to -π/6..+π/6)
    let mut delta = theta_v - theta_pos;
    // Wrap delta into (-π, π)
    while delta.to_f64() > core::f64::consts::PI {
        delta -= two_pi;
    }
    while delta.to_f64() < -core::f64::consts::PI {
        delta += two_pi;
    }

    let theta_out = theta_pos + blend * delta;

    // Amplitude: hexagon boundary at theta_out
    // For a hexagon with vertices at Vdc/2, the inscribed radius along any
    // angle within a 60° sector is:
    //   r(θ') = (Vdc/2) * cos(30°) / cos(θ' mod 60° - 30°)
    //         = (Vdc/2) * √3/2 / cos(θ' - sector_mid)
    let sqrt3_over2 = S::from_f64(3.0_f64.sqrt() / 2.0);
    let theta_sector = theta_out - theta_v; // angle relative to vertex
    let cos_sector = theta_sector.cos().abs().max(S::EPSILON);
    let r_hex = v_half * sqrt3_over2 / cos_sector;

    (r_hex * theta_out.cos(), r_hex * theta_out.sin())
}

/// Combined overmodulation: select Mode I or Mode II based on modulation index.
///
/// * MI ≤ 1.0: standard linear region (no clamp needed, but will clamp if exceeded).
/// * 1.0 < MI ≤ 4/π: Mode II applied.
///
/// The modulation index is computed from the input amplitude relative to
/// max_linear_voltage.
pub fn overmodulate<S: ControlScalar>(alpha: S, beta: S, v_dc: S, mi: S) -> (S, S) {
    let mi_max = S::from_f64(4.0 / core::f64::consts::PI);
    let mi_clamped = mi.clamp_val(S::ZERO, mi_max);

    if mi_clamped <= S::ONE {
        overmodulate_mode1(alpha, beta, v_dc)
    } else {
        overmodulate_mode2(alpha, beta, v_dc, mi_clamped)
    }
}

/// Overmodulation controller that automatically selects Mode I or II.
#[derive(Debug, Clone, Copy)]
pub struct OvermodulationController<S: ControlScalar> {
    /// DC bus voltage.
    pub v_dc: S,
    /// Modulation index threshold above which Mode II is used (typically 1.0).
    pub mi_threshold: S,
}

impl<S: ControlScalar> OvermodulationController<S> {
    pub fn new(v_dc: S) -> Self {
        Self {
            v_dc,
            mi_threshold: S::ONE,
        }
    }

    /// Compute the modulation index for a given αβ reference vector.
    ///
    /// MI = |Vαβ| / V_max_linear
    pub fn modulation_index(&self, alpha: S, beta: S) -> S {
        let mag = (alpha * alpha + beta * beta).sqrt();
        let v_max = max_linear_voltage(self.v_dc);
        if v_max > S::EPSILON {
            mag / v_max
        } else {
            S::ZERO
        }
    }

    /// Apply overmodulation to the reference vector.
    pub fn apply(&self, alpha: S, beta: S) -> (S, S) {
        let mi = self.modulation_index(alpha, beta);
        overmodulate(alpha, beta, self.v_dc, mi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_linear_voltage() {
        // For Vdc=400V: Vmax = 400/√3 ≈ 230.94V
        let v = max_linear_voltage(400.0_f32);
        let expected = 400.0_f32 / 3.0_f32.sqrt();
        assert!((v - expected).abs() < 0.01, "v={v}, expected={expected}");
    }

    #[test]
    fn test_mode1_clamping() {
        // Large vector should be clamped; small should pass unchanged.
        let v_dc = 400.0_f32;
        let v_max = max_linear_voltage(v_dc);

        // Small vector: no clamping
        let (a1, b1) = overmodulate_mode1(10.0_f32, 10.0_f32, v_dc);
        let mag1 = (a1 * a1 + b1 * b1).sqrt();
        assert!(mag1 <= (10.0_f32 * 2.0_f32.sqrt() + 0.01));

        // Large vector: clamped to v_max
        let big = v_max * 2.0;
        let (a2, b2) = overmodulate_mode1(big, 0.0_f32, v_dc);
        let mag2 = (a2 * a2 + b2 * b2).sqrt();
        assert!((mag2 - v_max).abs() < 0.01, "mag2={mag2}, v_max={v_max}");
    }

    #[test]
    fn test_overmodulation_controller_identity_below_threshold() {
        let ctrl = OvermodulationController::<f32>::new(400.0);
        // Vector within linear range: MI < 1
        let alpha = 100.0_f32;
        let beta = 0.0_f32;
        let mi = ctrl.modulation_index(alpha, beta);
        assert!(mi < 1.0, "Expected MI < 1, got {mi}");

        let (ao, bo) = ctrl.apply(alpha, beta);
        // Should remain essentially unchanged (Mode I does not change small vectors)
        assert!((ao - alpha).abs() < 0.01);
        assert!((bo - beta).abs() < 0.01);
    }

    #[test]
    fn test_max_six_step_voltage() {
        // For Vdc=400V: V6step = (2/π)*400 ≈ 254.65V
        let v = max_six_step_voltage(400.0_f32);
        let expected = 2.0_f32 / core::f32::consts::PI * 400.0;
        assert!((v - expected).abs() < 0.01, "v={v}");
    }
}
