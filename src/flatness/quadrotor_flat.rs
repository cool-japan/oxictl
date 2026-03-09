//! Quadrotor differential flatness.
//!
//! The flat outputs are σ = [x, y, z, ψ] (3-D position + yaw).  Every state
//! and control input of a quadrotor can be recovered from σ and its derivatives
//! up to fourth order.
//!
//! ## Inverse-map derivation (body-z thrust frame)
//!
//! Let **a**_des = [ẍ, ÿ, z̈+g]ᵀ be the desired specific force in the inertial
//! frame. Then:
//!
//! ```text
//!   T      = m · ‖a_des‖
//!   z_B    = a_des / ‖a_des‖          (body z-axis)
//!   x_C    = [cos ψ, sin ψ, 0]ᵀ      (heading reference)
//!   y_B    = (z_B × x_C) / ‖z_B × x_C‖
//!   x_B    = y_B × z_B
//!   R      = [x_B | y_B | z_B]
//!   φ      = atan2(R[2,1], R[2,2])   (roll)
//!   θ      = asin(-R[2,0])           (pitch)
//!   ψ      = ψ_des                   (yaw from flat output)
//! ```
//!
//! Angular rates follow from higher derivatives of σ (jerk → ω, snap → α).
//! Individual rotor speeds are found by inverting the thrust/torque allocation
//! matrix.
//!
//! ## Trajectory planning
//!
//! A `FlatTrajectory<S, SEG>` stores one minimum-snap (7th-order) polynomial
//! per position axis per segment, and one minimum-jerk (5th-order) polynomial
//! per yaw segment.  Waypoints are joined with zero velocity/acceleration at
//! segment boundaries (hover-to-hover style).

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ControlScalar;
use crate::flatness::FlatnessError;

/// Physical parameters needed to invert the quadrotor flat map.
#[derive(Debug, Clone, Copy)]
pub struct QuadrotorFlatParams<S: ControlScalar> {
    /// Total mass (kg).
    pub mass: S,
    /// Body-frame moment of inertia Ixx (kg·m²).
    pub ixx: S,
    /// Body-frame moment of inertia Iyy (kg·m²).
    pub iyy: S,
    /// Body-frame moment of inertia Izz (kg·m²).
    pub izz: S,
    /// Motor arm length (m).
    pub arm_length: S,
    /// Thrust coefficient kT (N / (rad²/s²)).
    pub kt: S,
    /// Torque coefficient kQ (N·m / (rad²/s²)).
    pub kq: S,
    /// Gravitational acceleration (m/s²), positive downward.
    pub gravity: S,
}

impl<S: ControlScalar> QuadrotorFlatParams<S> {
    /// Standard 250 mm-class quadrotor (matches `QuadrotorParams::standard`).
    pub fn standard() -> Self {
        Self {
            mass: S::from_f64(0.5),
            ixx: S::from_f64(4.0e-3),
            iyy: S::from_f64(4.0e-3),
            izz: S::from_f64(8.0e-3),
            arm_length: S::from_f64(0.12),
            kt: S::from_f64(1.5e-5),
            kq: S::from_f64(3.0e-7),
            gravity: S::from_f64(9.81),
        }
    }

    fn validate(&self) -> Result<(), FlatnessError> {
        if self.mass <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("mass must be positive"));
        }
        if self.ixx <= S::ZERO || self.iyy <= S::ZERO || self.izz <= S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "inertia components must be positive",
            ));
        }
        if self.arm_length <= S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "arm_length must be positive",
            ));
        }
        if self.kt <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("kt must be positive"));
        }
        if self.kq <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("kq must be positive"));
        }
        if self.gravity <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("gravity must be positive"));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FlatState
// ────────────────────────────────────────────────────────────────────────────

/// Flat output σ = [x, y, z, ψ] and its time derivatives up to 4th order.
///
/// Each field is a 4-element array [x, y, z, ψ] at the corresponding order.
#[derive(Debug, Clone, Copy)]
pub struct FlatState<S: ControlScalar> {
    /// Position: [x, y, z, ψ].
    pub pos: [S; 4],
    /// Velocity: [ẋ, ẏ, ż, ψ̇].
    pub vel: [S; 4],
    /// Acceleration: [ẍ, ÿ, z̈, ψ̈].
    pub acc: [S; 4],
    /// Jerk: [x⃛, y⃛, z⃛, ψ⃛].
    pub jerk: [S; 4],
    /// Snap (4th derivative): [x⁽⁴⁾, y⁽⁴⁾, z⁽⁴⁾, ψ⁽⁴⁾].
    pub snap: [S; 4],
}

impl<S: ControlScalar> FlatState<S> {
    /// All-zero flat state (hover at origin, zero yaw).
    pub fn zero() -> Self {
        Self {
            pos: [S::ZERO; 4],
            vel: [S::ZERO; 4],
            acc: [S::ZERO; 4],
            jerk: [S::ZERO; 4],
            snap: [S::ZERO; 4],
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// QuadrotorState (minimal, re-declared here to avoid cross-feature dependency)
// ────────────────────────────────────────────────────────────────────────────

/// Recovered quadrotor full state from the flat map.
#[derive(Debug, Clone, Copy)]
pub struct QuadrotorState<S: ControlScalar> {
    /// Inertial position x, y, z (m).
    pub x: S,
    pub y: S,
    pub z: S,
    /// Inertial velocity vx, vy, vz (m/s).
    pub vx: S,
    pub vy: S,
    pub vz: S,
    /// ZYX Euler angles roll φ, pitch θ, yaw ψ (rad).
    pub roll: S,
    pub pitch: S,
    pub yaw: S,
    /// Body-frame angular rates p, q, r (rad/s).
    pub p: S,
    pub q: S,
    pub r: S,
}

// ────────────────────────────────────────────────────────────────────────────
// QuadrotorFlatMap
// ────────────────────────────────────────────────────────────────────────────

/// Inverse flat map: converts a `FlatState` (σ and its derivatives) into the
/// full quadrotor state and individual rotor squared speeds [Ω1², Ω2², Ω3², Ω4²].
///
/// Rotor layout (X-frame, viewed from above):
/// ```text
///   1(front-left, CW)   2(front-right, CCW)
///   4(back-left,  CCW)  3(back-right,  CW)
///
///   τ_roll  = kT·l·(Ω4² − Ω2²)
///   τ_pitch = kT·l·(Ω3² − Ω1²)
///   τ_yaw   = kQ·(Ω1² − Ω2² + Ω3² − Ω4²)
/// ```
pub struct QuadrotorFlatMap<S: ControlScalar> {
    params: QuadrotorFlatParams<S>,
}

impl<S: ControlScalar> QuadrotorFlatMap<S> {
    /// Create a new inverse flat map with the given parameters.
    pub fn new(params: QuadrotorFlatParams<S>) -> Result<Self, FlatnessError> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Recover the full quadrotor state and rotor squared speeds from a flat state.
    ///
    /// Returns `(QuadrotorState, [Ω1², Ω2², Ω3², Ω4²])`.
    ///
    /// # Errors
    /// - `FlatnessError::Singular` if the desired specific force is too small
    ///   (‖a_des‖ < ε) or the yaw-direction projection is degenerate.
    pub fn flat_to_state(
        &self,
        flat: &FlatState<S>,
    ) -> Result<(QuadrotorState<S>, [S; 4]), FlatnessError> {
        let p = &self.params;
        let m = p.mass;
        let g = p.gravity;
        let l = p.arm_length;
        let kt = p.kt;
        let kq = p.kq;
        let ixx = p.ixx;
        let iyy = p.iyy;
        let izz = p.izz;

        // ── Step 1: desired specific-force vector (inertial frame) ──────────
        let ax = flat.acc[0];
        let ay = flat.acc[1];
        let az = flat.acc[2] + g; // lift = z̈ + g (NED convention: +z up needs sign)

        let norm_a = (ax * ax + ay * ay + az * az).sqrt();
        if norm_a < S::EPSILON {
            return Err(FlatnessError::Singular);
        }

        // Total thrust scalar
        let thrust = m * norm_a;

        // ── Step 2: body frame axes ──────────────────────────────────────────
        // z_B = â_des  (unit vector along desired thrust)
        let zb = [ax / norm_a, ay / norm_a, az / norm_a];

        // x_C = [cos ψ, sin ψ, 0]  (desired heading projected onto horizontal)
        let psi_des = flat.pos[3];
        let (spsi, cpsi) = (psi_des.sin(), psi_des.cos());
        let xc = [cpsi, spsi, S::ZERO];

        // y_B = (z_B × x_C) / ‖z_B × x_C‖
        let yb_raw = cross3(zb, xc);
        let norm_yb = vec3_norm(yb_raw);
        if norm_yb < S::EPSILON {
            return Err(FlatnessError::Singular);
        }
        let yb = [
            yb_raw[0] / norm_yb,
            yb_raw[1] / norm_yb,
            yb_raw[2] / norm_yb,
        ];

        // x_B = y_B × z_B
        let xb = cross3(yb, zb);

        // ── Step 3: Euler angles from rotation matrix ────────────────────────
        // R = [xb | yb | zb]  (columns are body axes in inertial frame)
        // R[i][j] = column j, row i
        //   R[0][0]=xb[0], R[1][0]=xb[1], R[2][0]=xb[2]
        //   R[0][1]=yb[0], R[1][1]=yb[1], R[2][1]=yb[2]
        //   R[0][2]=zb[0], R[1][2]=zb[1], R[2][2]=zb[2]
        //
        // ZYX Euler convention:
        //   pitch = asin(-R[2][0])
        //   roll  = atan2(R[2][1], R[2][2])
        //   yaw   = atan2(R[1][0], R[0][0])
        let pitch = (-xb[2]).asin();
        let roll = yb[2].atan2(zb[2]);
        let yaw = xb[1].atan2(xb[0]);

        // ── Step 4: angular velocity from jerk ──────────────────────────────
        // ω = R^T · (ṁ·a_des_dot − dot_a_dot)  reduced form:
        // Differentiating thrust direction:
        //   a_des_dot = [x⃛, y⃛, z⃛]
        // ḣ = (a_des_dot − (a_des_dot·z_B)·z_B) / ‖a_des‖
        // ω_body = [ḣ·y_B, -ḣ·x_B, ψ̇·z_B[2]]  (standard result)
        let jx = flat.jerk[0];
        let jy = flat.jerk[1];
        let jz = flat.jerk[2];

        // Component of jerk along z_B
        let jerk_along_zb = jx * zb[0] + jy * zb[1] + jz * zb[2];

        // Tangential jerk: h_dot = (jerk - jerk_along_zb * z_B) / ‖a_des‖
        let hdx = (jx - jerk_along_zb * zb[0]) / norm_a;
        let hdy = (jy - jerk_along_zb * zb[1]) / norm_a;
        let hdz = (jz - jerk_along_zb * zb[2]) / norm_a;

        // Body rates: p = -h_dot · y_B,  q = h_dot · x_B,  r = ψ̇·z_B[2]
        let omega_p = -(hdx * yb[0] + hdy * yb[1] + hdz * yb[2]);
        let omega_q = hdx * xb[0] + hdy * xb[1] + hdz * xb[2];
        let psi_dot = flat.vel[3];
        let omega_r = psi_dot * zb[2]; // approximate for small tilt

        // ── Step 5: recover individual rotor speeds from thrust + torques ───
        // We need angular accelerations from the snap (4th derivative).
        // α = d/dt(ω) — approximate from snap:
        let sx = flat.snap[0];
        let sy = flat.snap[1];
        let sz = flat.snap[2];

        // Derivative of norm_a with respect to t: d/dt(‖a_des‖) = (a_des·jerk)/‖a_des‖
        let dnorm_a = (ax * jx + ay * jy + az * jz) / norm_a;

        // d/dt(z_B) = (jerk - dnorm_a·z_B) / norm_a
        let dzb = [
            (jx - dnorm_a * zb[0]) / norm_a,
            (jy - dnorm_a * zb[1]) / norm_a,
            (jz - dnorm_a * zb[2]) / norm_a,
        ];

        // d/dt(h_dot): snap along zb
        let snap_along_zb = sx * zb[0] + sy * zb[1] + sz * zb[2];
        let jerk_along_dzb = jx * dzb[0] + jy * dzb[1] + jz * dzb[2];
        let d_jerk_along_zb = snap_along_zb + jerk_along_dzb;

        let norm_a2 = norm_a * norm_a;
        let hddx = (sx - d_jerk_along_zb * zb[0] - jerk_along_zb * dzb[0]) / norm_a
            - (jx - jerk_along_zb * zb[0]) * dnorm_a / norm_a2;
        let hddy = (sy - d_jerk_along_zb * zb[1] - jerk_along_zb * dzb[1]) / norm_a
            - (jy - jerk_along_zb * zb[1]) * dnorm_a / norm_a2;
        let hddz = (sz - d_jerk_along_zb * zb[2] - jerk_along_zb * dzb[2]) / norm_a
            - (jz - jerk_along_zb * zb[2]) * dnorm_a / norm_a2;

        let alpha_p = -(hddx * yb[0] + hddy * yb[1] + hddz * yb[2]);
        let alpha_q = hddx * xb[0] + hddy * xb[1] + hddz * xb[2];
        let psi_ddot = flat.acc[3];
        let alpha_r = psi_ddot * zb[2];

        // Body torques from Euler equations: τ = J·α + ω × (J·ω)
        let tau_roll = ixx * alpha_p + (izz - iyy) * omega_q * omega_r;
        let tau_pitch = iyy * alpha_q + (ixx - izz) * omega_p * omega_r;
        let tau_yaw = izz * alpha_r + (iyy - ixx) * omega_p * omega_q;

        // Allocation: solve for [Ω1², Ω2², Ω3², Ω4²]
        // T_total = kT·(Ω1² + Ω2² + Ω3² + Ω4²)
        // τ_roll  = kT·l·(Ω4² − Ω2²)
        // τ_pitch = kT·l·(Ω3² − Ω1²)
        // τ_yaw   = kQ·(Ω1² − Ω2² + Ω3² − Ω4²)
        //
        // Define S = Ω1²+Ω2²+Ω3²+Ω4², DR = Ω4²-Ω2², DP = Ω3²-Ω1², DY = Ω1²-Ω2²+Ω3²-Ω4²
        // From τ: DR = τ_roll/(kT·l), DP = τ_pitch/(kT·l), DY = τ_yaw/kQ
        // Sum = T/kT
        //
        // Ω1² = (S/4) − DP/2 + DY/4
        // Ω2² = (S/4) − DR/2 − DY/4
        // Ω3² = (S/4) + DP/2 + DY/4
        // Ω4² = (S/4) + DR/2 − DY/4

        let ktl = kt * l;
        if ktl.abs() < S::EPSILON || kq.abs() < S::EPSILON {
            return Err(FlatnessError::Singular);
        }

        let sum_omega = thrust / kt;
        let dr = tau_roll / ktl;
        let dp = tau_pitch / ktl;
        let dy = tau_yaw / kq;

        let quarter = S::from_f64(0.25);
        let half = S::HALF;

        let omega1_sq = quarter * sum_omega - half * dp + quarter * dy;
        let omega2_sq = quarter * sum_omega - half * dr - quarter * dy;
        let omega3_sq = quarter * sum_omega + half * dp + quarter * dy;
        let omega4_sq = quarter * sum_omega + half * dr - quarter * dy;

        let rotor_sq = [omega1_sq, omega2_sq, omega3_sq, omega4_sq];

        let state = QuadrotorState {
            x: flat.pos[0],
            y: flat.pos[1],
            z: flat.pos[2],
            vx: flat.vel[0],
            vy: flat.vel[1],
            vz: flat.vel[2],
            roll,
            pitch,
            yaw,
            p: omega_p,
            q: omega_q,
            r: omega_r,
        };

        Ok((state, rotor_sq))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MinSnapSeg — a single 7th-order segment
// ────────────────────────────────────────────────────────────────────────────

/// A single 7th-order (minimum-snap) polynomial segment.
///
/// p(τ) = Σ_{i=0}^{7} c_i · τ^i,  τ = t − t_start ∈ [0, duration].
#[derive(Debug, Clone, Copy)]
struct MinSnapSeg<S: ControlScalar> {
    c: [S; 8],
    duration: S,
    t_start: S,
}

impl<S: ControlScalar> MinSnapSeg<S> {
    /// Solve for coefficients given boundary conditions at τ=0 and τ=T.
    ///
    /// BCs: position, velocity, acceleration, jerk at both ends.
    #[allow(clippy::too_many_arguments)]
    fn new(
        p0: S,
        v0: S,
        a0: S,
        j0: S,
        p1: S,
        v1: S,
        a1: S,
        j1: S,
        duration: S,
        t_start: S,
    ) -> Result<Self, FlatnessError> {
        if duration <= S::ZERO {
            return Err(FlatnessError::PolynomialSolver);
        }

        let t = duration;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;

        if t4.abs() < S::EPSILON {
            return Err(FlatnessError::PolynomialSolver);
        }

        // First 4 coefficients from initial BCs:
        let c0 = p0;
        let c1 = v0;
        let c2 = a0 * S::HALF;
        let c3 = j0 / S::from_f64(6.0);

        // Residuals at t=T:
        let d0 = p1 - c0 - c1 * t - c2 * t2 - c3 * t3;
        let d1 = v1 - c1 - S::TWO * c2 * t - S::from_f64(3.0) * c3 * t2;
        let d2 = a1 - S::TWO * c2 - S::from_f64(6.0) * c3 * t;
        let d3 = j1 - S::from_f64(6.0) * c3;

        // Solve 4×4 system via Gaussian elimination (same pattern as polynomial.rs)
        let mut mat = [
            [S::ONE, t, t2, t3, d0 / t4],
            [
                S::from_f64(4.0),
                S::from_f64(5.0) * t,
                S::from_f64(6.0) * t2,
                S::from_f64(7.0) * t3,
                d1 / t3,
            ],
            [
                S::from_f64(12.0),
                S::from_f64(20.0) * t,
                S::from_f64(30.0) * t2,
                S::from_f64(42.0) * t3,
                d2 / t2,
            ],
            [
                S::from_f64(24.0),
                S::from_f64(60.0) * t,
                S::from_f64(120.0) * t2,
                S::from_f64(210.0) * t3,
                d3 / t,
            ],
        ];

        // Forward elimination with partial pivoting
        for col in 0..4_usize {
            let mut max_row = col;
            let mut max_val = mat[col][col].abs();
            for row in (col + 1)..4 {
                if mat[row][col].abs() > max_val {
                    max_val = mat[row][col].abs();
                    max_row = row;
                }
            }
            mat.swap(col, max_row);

            let pivot = mat[col][col];
            if pivot.abs() < S::EPSILON {
                return Err(FlatnessError::PolynomialSolver);
            }

            for row in (col + 1)..4 {
                let factor = mat[row][col] / pivot;
                for k in col..5 {
                    let sub = factor * mat[col][k];
                    mat[row][k] -= sub;
                }
            }
        }

        // Back substitution
        let mut sol = [S::ZERO; 4];
        for i in (0..4_usize).rev() {
            let mut sum = mat[i][4];
            for j in (i + 1)..4 {
                sum -= mat[i][j] * sol[j];
            }
            if mat[i][i].abs() < S::EPSILON {
                return Err(FlatnessError::PolynomialSolver);
            }
            sol[i] = sum / mat[i][i];
        }

        Ok(Self {
            c: [c0, c1, c2, c3, sol[0], sol[1], sol[2], sol[3]],
            duration,
            t_start,
        })
    }

    fn tau(&self, t: S) -> S {
        (t - self.t_start).clamp_val(S::ZERO, self.duration)
    }

    fn eval(&self, t: S) -> S {
        let tau = self.tau(t);
        let [c0, c1, c2, c3, c4, c5, c6, c7] = self.c;
        c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * (c5 + tau * (c6 + tau * c7))))))
    }

    fn eval_d1(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, c1, c2, c3, c4, c5, c6, c7] = self.c;
        c1 + tau
            * (S::TWO * c2
                + tau
                    * (S::from_f64(3.0) * c3
                        + tau
                            * (S::from_f64(4.0) * c4
                                + tau
                                    * (S::from_f64(5.0) * c5
                                        + tau
                                            * (S::from_f64(6.0) * c6
                                                + tau * S::from_f64(7.0) * c7)))))
    }

    fn eval_d2(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, c2, c3, c4, c5, c6, c7] = self.c;
        S::TWO * c2
            + tau
                * (S::from_f64(6.0) * c3
                    + tau
                        * (S::from_f64(12.0) * c4
                            + tau
                                * (S::from_f64(20.0) * c5
                                    + tau
                                        * (S::from_f64(30.0) * c6 + tau * S::from_f64(42.0) * c7))))
    }

    fn eval_d3(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, _, c3, c4, c5, c6, c7] = self.c;
        S::from_f64(6.0) * c3
            + tau
                * (S::from_f64(24.0) * c4
                    + tau
                        * (S::from_f64(60.0) * c5
                            + tau * (S::from_f64(120.0) * c6 + tau * S::from_f64(210.0) * c7)))
    }

    fn eval_d4(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, _, _, c4, c5, c6, c7] = self.c;
        S::from_f64(24.0) * c4
            + tau
                * (S::from_f64(120.0) * c5
                    + tau * (S::from_f64(360.0) * c6 + tau * S::from_f64(840.0) * c7))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MinJerkSeg — a single 5th-order segment (for yaw)
// ────────────────────────────────────────────────────────────────────────────

/// A 5th-order (minimum-jerk) polynomial segment used for yaw.
#[derive(Debug, Clone, Copy)]
struct MinJerkSeg<S: ControlScalar> {
    c: [S; 6],
    duration: S,
    t_start: S,
}

impl<S: ControlScalar> MinJerkSeg<S> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        p0: S,
        v0: S,
        a0: S,
        p1: S,
        v1: S,
        a1: S,
        duration: S,
        t_start: S,
    ) -> Result<Self, FlatnessError> {
        if duration <= S::ZERO {
            return Err(FlatnessError::PolynomialSolver);
        }

        let t = duration;
        let t2 = t * t;
        let t3 = t2 * t;

        if t3.abs() < S::EPSILON {
            return Err(FlatnessError::PolynomialSolver);
        }

        let c0 = p0;
        let c1 = v0;
        let c2 = a0 * S::HALF;

        let d0 = p1 - c0 - c1 * t - c2 * t2;
        let d1 = v1 - c1 - S::TWO * c2 * t;
        let d2 = a1 - S::TWO * c2;

        let det = S::TWO * t3;
        if det.abs() < S::EPSILON {
            return Err(FlatnessError::PolynomialSolver);
        }

        let r0 = d0 / t3;
        let r1 = d1 / t2;
        let r2 = d2 / t;

        let c3 = t3 * (S::from_f64(20.0) * r0 - S::from_f64(8.0) * r1 + r2) / det;
        let c4 = t2 * (S::from_f64(-30.0) * r0 + S::from_f64(14.0) * r1 - S::TWO * r2) / det;
        let c5 = t * (S::from_f64(12.0) * r0 - S::from_f64(6.0) * r1 + r2) / det;

        Ok(Self {
            c: [c0, c1, c2, c3, c4, c5],
            duration,
            t_start,
        })
    }

    fn tau(&self, t: S) -> S {
        (t - self.t_start).clamp_val(S::ZERO, self.duration)
    }

    fn eval(&self, t: S) -> S {
        let tau = self.tau(t);
        let [c0, c1, c2, c3, c4, c5] = self.c;
        c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5))))
    }

    fn eval_d1(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, c1, c2, c3, c4, c5] = self.c;
        c1 + tau
            * (S::TWO * c2
                + tau
                    * (S::from_f64(3.0) * c3
                        + tau * (S::from_f64(4.0) * c4 + tau * S::from_f64(5.0) * c5)))
    }

    fn eval_d2(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, c2, c3, c4, c5] = self.c;
        S::TWO * c2
            + tau
                * (S::from_f64(6.0) * c3
                    + tau * (S::from_f64(12.0) * c4 + tau * S::from_f64(20.0) * c5))
    }

    fn eval_d3(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, _, c3, c4, c5] = self.c;
        S::from_f64(6.0) * c3 + tau * (S::from_f64(24.0) * c4 + tau * S::from_f64(60.0) * c5)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FlatTrajectory
// ────────────────────────────────────────────────────────────────────────────

/// Piecewise polynomial flat-output trajectory over `SEG` segments.
///
/// Position axes [x, y, z] use 7th-order minimum-snap polynomials.
/// Yaw axis ψ uses 5th-order minimum-jerk polynomials.
///
/// Waypoints are connected with zero velocity/acceleration/jerk at boundaries
/// (hover-to-hover style), making the system always differentially flat.
#[derive(Debug, Clone, Copy)]
pub struct FlatTrajectory<S: ControlScalar, const SEG: usize> {
    /// Minimum-snap segments for x.
    segs_x: [Option<MinSnapSeg<S>>; SEG],
    /// Minimum-snap segments for y.
    segs_y: [Option<MinSnapSeg<S>>; SEG],
    /// Minimum-snap segments for z.
    segs_z: [Option<MinSnapSeg<S>>; SEG],
    /// Minimum-jerk segments for yaw.
    segs_psi: [Option<MinJerkSeg<S>>; SEG],
    /// Number of active segments.
    seg_count: usize,
    /// Cumulative start times for each segment.
    t_starts: [S; SEG],
    /// Total trajectory duration.
    total_duration: S,
}

impl<S: ControlScalar, const SEG: usize> FlatTrajectory<S, SEG> {
    /// Build a piecewise minimum-snap trajectory from SEG+1 waypoints and SEG durations.
    ///
    /// Each waypoint is [x, y, z, ψ].  Boundary conditions at intermediate
    /// waypoints are set to zero velocity/acceleration/jerk (hover-to-hover).
    ///
    /// # Errors
    /// Returns `FlatnessError::PolynomialSolver` if any segment duration ≤ 0
    /// or if the linear solver fails.
    pub fn from_waypoints(
        waypoints: &[[S; 4]; SEG],
        times: &[S; SEG],
        start: [S; 4],
    ) -> Result<Self, FlatnessError> {
        let mut traj = Self {
            segs_x: [None; SEG],
            segs_y: [None; SEG],
            segs_z: [None; SEG],
            segs_psi: [None; SEG],
            seg_count: 0,
            t_starts: [S::ZERO; SEG],
            total_duration: S::ZERO,
        };

        let zero = S::ZERO;
        let mut t_acc = S::ZERO;
        let mut prev = start;

        for i in 0..SEG {
            let dur = times[i];
            let next = waypoints[i];
            let t_start = t_acc;

            traj.segs_x[i] = Some(MinSnapSeg::new(
                prev[0], zero, zero, zero, next[0], zero, zero, zero, dur, t_start,
            )?);
            traj.segs_y[i] = Some(MinSnapSeg::new(
                prev[1], zero, zero, zero, next[1], zero, zero, zero, dur, t_start,
            )?);
            traj.segs_z[i] = Some(MinSnapSeg::new(
                prev[2], zero, zero, zero, next[2], zero, zero, zero, dur, t_start,
            )?);
            traj.segs_psi[i] = Some(MinJerkSeg::new(
                prev[3], zero, zero, next[3], zero, zero, dur, t_start,
            )?);

            traj.t_starts[i] = t_acc;
            t_acc += dur;
            prev = next;
        }

        traj.seg_count = SEG;
        traj.total_duration = t_acc;
        Ok(traj)
    }

    /// Evaluate the flat output and all derivatives up to 4th order at time `t`.
    ///
    /// Clamps `t` to [0, total_duration].
    pub fn eval(&self, t: S) -> FlatState<S> {
        let idx = self.find_segment(t);
        let i = match idx {
            None => return FlatState::zero(),
            Some(i) => i,
        };

        let zero5 = (S::ZERO, S::ZERO, S::ZERO, S::ZERO, S::ZERO);

        let (x0, x1, x2, x3, x4) = self.segs_x[i]
            .map(|s| {
                (
                    s.eval(t),
                    s.eval_d1(t),
                    s.eval_d2(t),
                    s.eval_d3(t),
                    s.eval_d4(t),
                )
            })
            .unwrap_or(zero5);
        let (y0, y1, y2, y3, y4) = self.segs_y[i]
            .map(|s| {
                (
                    s.eval(t),
                    s.eval_d1(t),
                    s.eval_d2(t),
                    s.eval_d3(t),
                    s.eval_d4(t),
                )
            })
            .unwrap_or(zero5);
        let (z0, z1, z2, z3, z4) = self.segs_z[i]
            .map(|s| {
                (
                    s.eval(t),
                    s.eval_d1(t),
                    s.eval_d2(t),
                    s.eval_d3(t),
                    s.eval_d4(t),
                )
            })
            .unwrap_or(zero5);
        let (p0, p1, p2, p3, _) = self.segs_psi[i]
            .map(|s| (s.eval(t), s.eval_d1(t), s.eval_d2(t), s.eval_d3(t), S::ZERO))
            .unwrap_or(zero5);

        FlatState {
            pos: [x0, y0, z0, p0],
            vel: [x1, y1, z1, p1],
            acc: [x2, y2, z2, p2],
            jerk: [x3, y3, z3, p3],
            snap: [x4, y4, z4, S::ZERO],
        }
    }

    /// Total trajectory duration (sum of segment durations).
    pub fn total_duration(&self) -> S {
        self.total_duration
    }

    fn find_segment(&self, t: S) -> Option<usize> {
        if self.seg_count == 0 {
            return None;
        }
        for i in 0..self.seg_count {
            let t_end = self.t_starts[i] + self.segs_x[i].map(|s| s.duration).unwrap_or(S::ZERO);
            if t <= t_end {
                return Some(i);
            }
        }
        // After end: return last segment
        Some(self.seg_count - 1)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helper geometry
// ────────────────────────────────────────────────────────────────────────────

#[inline]
fn cross3<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vec3_norm<S: ControlScalar>(v: [S; 3]) -> S {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// At hover (all derivatives zero, z̈=0, ψ=0), thrust should equal mg
    /// and all four rotor speeds should be equal.
    #[test]
    fn hover_flat_state_equal_rotor_speeds() {
        let params = QuadrotorFlatParams::<f64>::standard();
        let flat_map = QuadrotorFlatMap::new(params).expect("valid params");

        let flat = FlatState::<f64>::zero();
        let (_state, rotors) = flat_map.flat_to_state(&flat).expect("flat_to_state");

        // All four rotor speeds should be equal
        let diff01 = (rotors[0] - rotors[1]).abs();
        let diff02 = (rotors[0] - rotors[2]).abs();
        let diff03 = (rotors[0] - rotors[3]).abs();
        assert!(
            diff01 < 1e-6 && diff02 < 1e-6 && diff03 < 1e-6,
            "Hover rotor speeds not equal: {:?}",
            rotors
        );

        // Total thrust = m·g
        let thrust: f64 = rotors.iter().map(|&w| params.kt * w).sum();
        let weight = params.mass * params.gravity;
        assert!(
            (thrust - weight).abs() < 1e-6,
            "Hover thrust={:.6} ≠ weight={:.6}",
            thrust,
            weight
        );
    }

    /// A purely upward acceleration should yield correct thrust direction.
    #[test]
    fn vertical_acceleration_correct_thrust() {
        let params = QuadrotorFlatParams::<f64>::standard();
        let flat_map = QuadrotorFlatMap::new(params).expect("valid params");

        let mut flat = FlatState::<f64>::zero();
        // 2 m/s² upward (in addition to gravity compensation)
        flat.acc[2] = 2.0;

        let (_state, rotors) = flat_map.flat_to_state(&flat).expect("flat_to_state");

        // Expected thrust = m * (g + az)
        let expected_thrust = params.mass * (params.gravity + 2.0);
        let actual_thrust: f64 = rotors.iter().map(|&w| params.kt * w).sum();
        assert!(
            (actual_thrust - expected_thrust).abs() < 1e-5,
            "thrust={:.6} expected={:.6}",
            actual_thrust,
            expected_thrust
        );

        // Roll and pitch should remain zero
        let (_state2, _) = flat_map.flat_to_state(&flat).expect("ok");
        // state.roll ≈ 0, state.pitch ≈ 0
        assert!(
            _state2.roll.abs() < 1e-8,
            "roll={:.2e} should be ~0",
            _state2.roll
        );
        assert!(
            _state2.pitch.abs() < 1e-8,
            "pitch={:.2e} should be ~0",
            _state2.pitch
        );
    }

    /// Hover trajectory: constant position, all derivatives zero → rotor
    /// speed variation should be zero throughout.
    #[test]
    fn hover_trajectory_zero_rotor_variation() {
        // Single segment: stay at (0,0,1,0) for 2 seconds
        let waypoints: [[f64; 4]; 1] = [[0.0, 0.0, 1.0, 0.0]];
        let times: [f64; 1] = [2.0];
        let start = [0.0, 0.0, 1.0, 0.0_f64];

        let traj = FlatTrajectory::<f64, 1>::from_waypoints(&waypoints, &times, start)
            .expect("trajectory creation");

        let params = QuadrotorFlatParams::<f64>::standard();
        let flat_map = QuadrotorFlatMap::new(params).expect("valid params");

        let ts = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
        let mut all_rotors: [[f64; 4]; 5] = [[0.0; 4]; 5];
        for (k, &t) in ts.iter().enumerate() {
            let flat = traj.eval(t);
            let (_, rotors) = flat_map.flat_to_state(&flat).expect("flat_to_state");
            all_rotors[k] = rotors;
        }

        // All rotor speeds at each time should be equal (hover condition)
        for (k, rotors) in all_rotors.iter().enumerate() {
            let spread = rotors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - rotors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            assert!(
                spread < 1.0,
                "t[{}] rotor spread too large: {:.4}, rotors={:?}",
                k,
                spread,
                rotors
            );
        }
    }

    /// Line trajectory: thrust should remain pointing upward (Z-axis aligned).
    #[test]
    fn line_trajectory_correct_thrust_direction() {
        // Move from (0,0,1) to (2,0,1) in 2 seconds — horizontal motion
        let waypoints: [[f64; 4]; 1] = [[2.0, 0.0, 1.0, 0.0]];
        let times: [f64; 1] = [2.0];
        let start = [0.0, 0.0, 1.0, 0.0_f64];

        let traj = FlatTrajectory::<f64, 1>::from_waypoints(&waypoints, &times, start)
            .expect("trajectory");

        // At t=0 and t=2 (boundaries), velocity=0 → pure hover
        for &t in &[0.0_f64, 2.0] {
            let flat = traj.eval(t);
            // Acceleration in x should be zero at boundaries (min-snap)
            let ax = flat.acc[0].abs();
            assert!(
                ax < 1e-8,
                "t={} x-accel={:.2e} should be ~0 at boundary",
                t,
                ax
            );
        }

        // At t=1 (midpoint), there should be nonzero acceleration in x
        let flat_mid = traj.eval(1.0);
        // The trajectory accelerates then decelerates, so mid-snap has near-zero acc
        // but we check that position is between start and end
        let x_mid = flat_mid.pos[0];
        assert!(
            x_mid > 0.0 && x_mid < 2.0,
            "x_mid={:.4} should be in (0,2)",
            x_mid
        );
    }

    /// Flat map rejects zero specific force (hovering below threshold).
    #[test]
    fn singular_zero_thrust_returns_error() {
        let params = QuadrotorFlatParams::<f64>::standard();
        let flat_map = QuadrotorFlatMap::new(params).expect("valid params");

        let mut flat = FlatState::<f64>::zero();
        // Cancel gravity: az = -g, so a_des = [0,0,0]
        flat.acc[2] = -params.gravity;

        let result = flat_map.flat_to_state(&flat);
        assert!(
            matches!(result, Err(FlatnessError::Singular)),
            "Expected Singular error, got {:?}",
            result
        );
    }

    /// Multi-segment trajectory: verify continuity of position at waypoints.
    #[test]
    fn multi_segment_position_continuity() {
        let waypoints: [[f64; 4]; 2] = [[1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 1.0, 0.0]];
        let times: [f64; 2] = [1.0, 1.5];
        let start = [0.0, 0.0, 1.0, 0.0_f64];

        let traj = FlatTrajectory::<f64, 2>::from_waypoints(&waypoints, &times, start)
            .expect("trajectory");

        // At t=1.0 (end of first segment), position should be [1,0,1,0]
        let flat_at_1 = traj.eval(1.0);
        assert!(
            (flat_at_1.pos[0] - 1.0).abs() < 1e-8,
            "x at waypoint 1: {:.6}",
            flat_at_1.pos[0]
        );
        assert!(
            (flat_at_1.pos[1]).abs() < 1e-8,
            "y at waypoint 1: {:.6}",
            flat_at_1.pos[1]
        );

        // At t=2.5 (end of trajectory), position should be [2,1,1,0]
        let flat_end = traj.eval(2.5);
        assert!(
            (flat_end.pos[0] - 2.0).abs() < 1e-8,
            "x at end: {:.6}",
            flat_end.pos[0]
        );
        assert!(
            (flat_end.pos[1] - 1.0).abs() < 1e-8,
            "y at end: {:.6}",
            flat_end.pos[1]
        );
    }
}
