//! Diagnostic coverage monitor and fault tree analysis.
//!
//! Provides:
//! - [`DiagnosticMonitor`] — tracks detected vs. undetected dangerous failures
//!   and computes DC and SFF at runtime.
//! - [`ReliabilityBlock`] — static block data: failure rate, SFF, DC.
//! - [`FaultTreeNode`] — recursive fault tree structure (AND/OR gates +
//!   basic events) with a `compute_pfd` method for top-event PFD.
//!
//! All arithmetic uses `f64` to preserve the precision required for
//! IEC 61508 safety calculations.  The implementation is `no_std` compatible.

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// DiagnosticCoverageValue / SafeFailureFraction
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic Coverage (DC) as a floating-point fraction in [0.0, 1.0].
///
/// DC = λ_detected / λ_total
///
/// where λ_detected is the rate of dangerous failures caught by diagnostics
/// and λ_total is the total dangerous failure rate.
///
/// A higher DC means a greater proportion of failures are caught early.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DiagnosticCoverageValue(f64);

impl DiagnosticCoverageValue {
    /// Construct from a pre-computed fraction.
    ///
    /// Returns `Err` if `dc` is not in [0.0, 1.0].
    pub fn new(dc: f64) -> Result<Self, DiagnosticError> {
        if dc.is_nan() || !(0.0..=1.0).contains(&dc) {
            Err(DiagnosticError::InvalidFraction { value: dc })
        } else {
            Ok(Self(dc))
        }
    }

    /// The raw fraction value.
    pub fn fraction(self) -> f64 {
        self.0
    }

    /// Classify into the IEC 61508 DC tier.
    ///
    /// | Tier      | DC range       |
    /// |-----------|----------------|
    /// | Low       | DC < 60 %      |
    /// | Medium    | 60 % ≤ DC < 90 %|
    /// | High      | 90 % ≤ DC < 99 %|
    /// | Very High | DC ≥ 99 %      |
    pub fn tier(self) -> DcTier {
        DcTier::from_fraction(self.0)
    }
}

/// IEC 61508 diagnostic coverage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DcTier {
    /// DC < 60 %
    Low,
    /// 60 % ≤ DC < 90 %
    Medium,
    /// 90 % ≤ DC < 99 %
    High,
    /// DC ≥ 99 %
    VeryHigh,
}

impl DcTier {
    /// Classify a raw DC fraction.
    pub fn from_fraction(dc: f64) -> Self {
        if dc >= 0.99 {
            DcTier::VeryHigh
        } else if dc >= 0.90 {
            DcTier::High
        } else if dc >= 0.60 {
            DcTier::Medium
        } else {
            DcTier::Low
        }
    }
}

/// Safe Failure Fraction (SFF) as a floating-point fraction in [0.0, 1.0].
///
/// SFF = (λ_safe + λ_dd) / λ_total
///
/// where:
/// - λ_safe = safe (non-dangerous) failure rate
/// - λ_dd   = detected dangerous failure rate
/// - λ_total = total failure rate
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SafeFailureFraction(f64);

impl SafeFailureFraction {
    /// Construct from a pre-computed fraction.
    ///
    /// Returns `Err` if `sff` is not in [0.0, 1.0].
    pub fn new(sff: f64) -> Result<Self, DiagnosticError> {
        if sff.is_nan() || !(0.0..=1.0).contains(&sff) {
            Err(DiagnosticError::InvalidFraction { value: sff })
        } else {
            Ok(Self(sff))
        }
    }

    /// The raw SFF value.
    pub fn fraction(self) -> f64 {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReliabilityBlock
// ─────────────────────────────────────────────────────────────────────────────

/// Static reliability data for a single hardware subsystem or component.
///
/// Used as an input to fault tree computations and architectural constraints
/// analysis (IEC 61508 Part 2, Table 3).
#[derive(Debug, Clone, Copy)]
pub struct ReliabilityBlock {
    /// Total failure rate λ (failures per hour).
    pub failure_rate_per_hour: f64,
    /// Fraction of total failures that are safe (including detected dangerous).
    /// SFF = (λ_safe + λ_dd) / λ_total.
    pub safe_failure_fraction: f64,
    /// Fraction of dangerous failures detected by diagnostics.
    /// DC = λ_dd / (λ_dd + λ_du).
    pub diagnostic_coverage: f64,
}

impl ReliabilityBlock {
    /// Construct a new block, validating that fractions are in [0, 1] and
    /// the failure rate is non-negative.
    pub fn new(
        failure_rate_per_hour: f64,
        safe_failure_fraction: f64,
        diagnostic_coverage: f64,
    ) -> Result<Self, DiagnosticError> {
        if failure_rate_per_hour < 0.0 || !failure_rate_per_hour.is_finite() {
            return Err(DiagnosticError::InvalidFailureRate {
                value: failure_rate_per_hour,
            });
        }
        if !(0.0..=1.0).contains(&safe_failure_fraction) || safe_failure_fraction.is_nan() {
            return Err(DiagnosticError::InvalidFraction {
                value: safe_failure_fraction,
            });
        }
        if !(0.0..=1.0).contains(&diagnostic_coverage) || diagnostic_coverage.is_nan() {
            return Err(DiagnosticError::InvalidFraction {
                value: diagnostic_coverage,
            });
        }
        Ok(Self {
            failure_rate_per_hour,
            safe_failure_fraction,
            diagnostic_coverage,
        })
    }

    /// Dangerous failure rate: λ_d = λ_total * (1 − SFF).
    pub fn lambda_dangerous(&self) -> f64 {
        self.failure_rate_per_hour * (1.0 - self.safe_failure_fraction)
    }

    /// Detected dangerous failure rate: λ_dd = λ_d * DC.
    pub fn lambda_detected_dangerous(&self) -> f64 {
        self.lambda_dangerous() * self.diagnostic_coverage
    }

    /// Undetected dangerous failure rate: λ_du = λ_d * (1 − DC).
    pub fn lambda_undetected_dangerous(&self) -> f64 {
        self.lambda_dangerous() * (1.0 - self.diagnostic_coverage)
    }

    /// PFD_avg for low-demand mode: PFD = λ_du * T_proof / 2.
    pub fn pfd_avg(&self, proof_test_interval_hours: f64) -> f64 {
        self.lambda_undetected_dangerous() * proof_test_interval_hours * 0.5
    }

    /// PFH for continuous mode: PFH = λ_du.
    pub fn pfh(&self) -> f64 {
        self.lambda_undetected_dangerous()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FaultTree  (flat arena, index-based — avoids recursive enum infinite size)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum children an AND or OR gate may have.
pub const MAX_CHILDREN: usize = 8;

/// Maximum number of nodes a [`FaultTree`] arena can hold.
pub const MAX_TREE_NODES: usize = 32;

/// Internal node kind stored in the flat arena.
#[derive(Debug, Clone, Copy)]
enum NodeKind {
    /// Basic event: constant failure rate with optional common-cause factor.
    BasicEvent { lambda_per_hour: f64, beta: f64 },
    /// AND gate: PFD = product of children PFDs.
    And {
        child_indices: [u8; MAX_CHILDREN],
        child_count: u8,
    },
    /// OR gate: PFD = 1 − product of (1 − PFD_i).
    Or {
        child_indices: [u8; MAX_CHILDREN],
        child_count: u8,
    },
}

/// A node descriptor stored inside a [`FaultTree`] arena.
#[derive(Debug, Clone, Copy)]
pub struct FaultTreeNode {
    kind: NodeKind,
}

/// A flat-arena fault tree that avoids recursive data structures.
///
/// Nodes are stored in a fixed-size array; references between nodes use
/// `u8` indices.  The root is always the last inserted node.
///
/// PFD computation follows IEC 61508 Part 6 Annex B:
/// - Basic event: PFD = λ * (1 − β) * T_proof / 2
/// - AND gate:    PFD = Π PFD_i
/// - OR gate:     PFD = 1 − Π (1 − PFD_i)
#[derive(Debug, Clone, Copy)]
pub struct FaultTree {
    nodes: [FaultTreeNode; MAX_TREE_NODES],
    count: usize,
}

impl FaultTree {
    /// Create an empty fault tree.
    pub fn new() -> Self {
        Self {
            nodes: [FaultTreeNode {
                kind: NodeKind::BasicEvent {
                    lambda_per_hour: 0.0,
                    beta: 0.0,
                },
            }; MAX_TREE_NODES],
            count: 0,
        }
    }

    /// Add a basic event node.  Returns its index, or `Err` if the arena is full.
    pub fn add_basic_event(
        &mut self,
        lambda_per_hour: f64,
        beta: f64,
    ) -> Result<u8, DiagnosticError> {
        self.push(FaultTreeNode {
            kind: NodeKind::BasicEvent {
                lambda_per_hour,
                beta,
            },
        })
    }

    /// Add an AND gate with the given child indices.
    pub fn add_and(&mut self, children: &[u8]) -> Result<u8, DiagnosticError> {
        if children.len() > MAX_CHILDREN {
            return Err(DiagnosticError::TooManyChildren {
                count: children.len(),
                max: MAX_CHILDREN,
            });
        }
        let mut idx = [0u8; MAX_CHILDREN];
        for (i, &c) in children.iter().enumerate() {
            idx[i] = c;
        }
        self.push(FaultTreeNode {
            kind: NodeKind::And {
                child_indices: idx,
                child_count: children.len() as u8,
            },
        })
    }

    /// Add an OR gate with the given child indices.
    pub fn add_or(&mut self, children: &[u8]) -> Result<u8, DiagnosticError> {
        if children.len() > MAX_CHILDREN {
            return Err(DiagnosticError::TooManyChildren {
                count: children.len(),
                max: MAX_CHILDREN,
            });
        }
        let mut idx = [0u8; MAX_CHILDREN];
        for (i, &c) in children.iter().enumerate() {
            idx[i] = c;
        }
        self.push(FaultTreeNode {
            kind: NodeKind::Or {
                child_indices: idx,
                child_count: children.len() as u8,
            },
        })
    }

    /// Compute the top-event PFD from a given root node index.
    ///
    /// Returns `Err(DiagnosticError::InvalidTestInterval)` if `test_interval_hours`
    /// is negative or non-finite.
    pub fn compute_pfd(&self, root: u8, test_interval_hours: f64) -> Result<f64, DiagnosticError> {
        if !test_interval_hours.is_finite() || test_interval_hours < 0.0 {
            return Err(DiagnosticError::InvalidTestInterval {
                value: test_interval_hours,
            });
        }
        Ok(self.pfd_for(root as usize, test_interval_hours))
    }

    fn pfd_for(&self, idx: usize, ti: f64) -> f64 {
        if idx >= self.count {
            return 0.0;
        }
        match self.nodes[idx].kind {
            NodeKind::BasicEvent {
                lambda_per_hour,
                beta,
            } => {
                let eff = lambda_per_hour * (1.0 - beta.clamp(0.0, 1.0));
                eff * ti * 0.5
            }
            NodeKind::And {
                child_indices,
                child_count,
            } => {
                let mut product = 1.0_f64;
                for &ci in child_indices.iter().take(child_count as usize) {
                    product *= self.pfd_for(ci as usize, ti);
                }
                product
            }
            NodeKind::Or {
                child_indices,
                child_count,
            } => {
                let mut complement = 1.0_f64;
                for &ci in child_indices.iter().take(child_count as usize) {
                    complement *= 1.0 - self.pfd_for(ci as usize, ti);
                }
                1.0 - complement
            }
        }
    }

    fn push(&mut self, node: FaultTreeNode) -> Result<u8, DiagnosticError> {
        if self.count >= MAX_TREE_NODES {
            return Err(DiagnosticError::TooManyChildren {
                count: self.count + 1,
                max: MAX_TREE_NODES,
            });
        }
        let idx = self.count;
        self.nodes[idx] = node;
        self.count += 1;
        Ok(idx as u8)
    }

    /// Number of nodes in the arena.
    pub fn node_count(&self) -> usize {
        self.count
    }
}

impl Default for FaultTree {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DiagnosticMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime accumulator for dangerous failure rates, used to compute DC and SFF.
///
/// Each call to `record_detected_failure` or `record_undetected_failure` adds
/// the supplied failure rate λ to the appropriate accumulator.  The total safe
/// failure rate is tracked separately via `record_safe_failure`.
///
/// DC  = λ_dd / (λ_dd + λ_du)
/// SFF = (λ_safe + λ_dd) / (λ_safe + λ_dd + λ_du)
///
/// where λ_dd = detected dangerous, λ_du = undetected dangerous.
#[derive(Debug, Clone, Copy)]
pub struct DiagnosticMonitor {
    /// Accumulated detected-dangerous failure rate (h⁻¹).
    lambda_detected: f64,
    /// Accumulated undetected-dangerous failure rate (h⁻¹).
    lambda_undetected: f64,
    /// Accumulated safe failure rate (h⁻¹).
    lambda_safe: f64,
}

impl DiagnosticMonitor {
    /// Create a new monitor with all accumulators at zero.
    pub fn new() -> Self {
        Self {
            lambda_detected: 0.0,
            lambda_undetected: 0.0,
            lambda_safe: 0.0,
        }
    }

    /// Record a detected dangerous failure at rate `lambda` (h⁻¹).
    ///
    /// Returns `Err` if `lambda` is negative or non-finite.
    pub fn record_detected_failure(&mut self, lambda: f64) -> Result<(), DiagnosticError> {
        validate_rate(lambda)?;
        self.lambda_detected += lambda;
        Ok(())
    }

    /// Record an undetected dangerous failure at rate `lambda` (h⁻¹).
    ///
    /// Returns `Err` if `lambda` is negative or non-finite.
    pub fn record_undetected_failure(&mut self, lambda: f64) -> Result<(), DiagnosticError> {
        validate_rate(lambda)?;
        self.lambda_undetected += lambda;
        Ok(())
    }

    /// Record a safe (non-dangerous) failure at rate `lambda` (h⁻¹).
    ///
    /// Returns `Err` if `lambda` is negative or non-finite.
    pub fn record_safe_failure(&mut self, lambda: f64) -> Result<(), DiagnosticError> {
        validate_rate(lambda)?;
        self.lambda_safe += lambda;
        Ok(())
    }

    /// Diagnostic Coverage: λ_dd / (λ_dd + λ_du).
    ///
    /// Returns 0.0 if no dangerous failures have been recorded.
    pub fn diagnostic_coverage(&self) -> f64 {
        let total_dangerous = self.lambda_detected + self.lambda_undetected;
        if total_dangerous <= 0.0 {
            0.0
        } else {
            self.lambda_detected / total_dangerous
        }
    }

    /// Safe Failure Fraction: (λ_safe + λ_dd) / λ_total.
    ///
    /// Returns 1.0 if no failures have been recorded (vacuously safe).
    pub fn safe_failure_fraction(&self) -> f64 {
        let total = self.lambda_safe + self.lambda_detected + self.lambda_undetected;
        if total <= 0.0 {
            1.0 // vacuously safe — no failures recorded
        } else {
            (self.lambda_safe + self.lambda_detected) / total
        }
    }

    /// Total dangerous failure rate (λ_dd + λ_du).
    pub fn lambda_dangerous_total(&self) -> f64 {
        self.lambda_detected + self.lambda_undetected
    }

    /// Total failure rate (λ_safe + λ_dd + λ_du).
    pub fn lambda_total(&self) -> f64 {
        self.lambda_safe + self.lambda_detected + self.lambda_undetected
    }

    /// Detected dangerous failure rate accumulated so far.
    pub fn lambda_detected(&self) -> f64 {
        self.lambda_detected
    }

    /// Undetected dangerous failure rate accumulated so far.
    pub fn lambda_undetected(&self) -> f64 {
        self.lambda_undetected
    }

    /// Reset all accumulators to zero.
    pub fn reset(&mut self) {
        self.lambda_detected = 0.0;
        self.lambda_undetected = 0.0;
        self.lambda_safe = 0.0;
    }
}

impl Default for DiagnosticMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from diagnostic and fault-tree operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiagnosticError {
    /// A probability or fractional value was outside [0, 1].
    InvalidFraction { value: f64 },
    /// A failure rate was negative or non-finite.
    InvalidFailureRate { value: f64 },
    /// A proof-test interval was negative or non-finite.
    InvalidTestInterval { value: f64 },
    /// Too many children for a fault-tree gate.
    TooManyChildren { count: usize, max: usize },
}

impl core::fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiagnosticError::InvalidFraction { value } => {
                write!(f, "fraction {value} is not in [0, 1]")
            }
            DiagnosticError::InvalidFailureRate { value } => {
                write!(f, "failure rate {value} is negative or non-finite")
            }
            DiagnosticError::InvalidTestInterval { value } => {
                write!(f, "test interval {value} is negative or non-finite")
            }
            DiagnosticError::TooManyChildren { count, max } => {
                write!(f, "gate has {count} children but maximum is {max}")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rate(lambda: f64) -> Result<(), DiagnosticError> {
    if !lambda.is_finite() || lambda < 0.0 {
        Err(DiagnosticError::InvalidFailureRate { value: lambda })
    } else {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DiagnosticCoverageValue ─────────────────────────────────────────────

    #[test]
    fn dc_value_valid_range() {
        assert!(DiagnosticCoverageValue::new(0.0).is_ok());
        assert!(DiagnosticCoverageValue::new(1.0).is_ok());
        assert!(DiagnosticCoverageValue::new(0.95).is_ok());
    }

    #[test]
    fn dc_value_invalid_range() {
        assert!(matches!(
            DiagnosticCoverageValue::new(-0.1),
            Err(DiagnosticError::InvalidFraction { .. })
        ));
        assert!(matches!(
            DiagnosticCoverageValue::new(1.1),
            Err(DiagnosticError::InvalidFraction { .. })
        ));
        assert!(matches!(
            DiagnosticCoverageValue::new(f64::NAN),
            Err(DiagnosticError::InvalidFraction { .. })
        ));
    }

    #[test]
    fn dc_tier_classification() {
        assert_eq!(DcTier::from_fraction(0.30), DcTier::Low);
        assert_eq!(DcTier::from_fraction(0.75), DcTier::Medium);
        assert_eq!(DcTier::from_fraction(0.95), DcTier::High);
        assert_eq!(DcTier::from_fraction(0.995), DcTier::VeryHigh);
        // Boundary: exactly 0.99 → VeryHigh
        assert_eq!(DcTier::from_fraction(0.99), DcTier::VeryHigh);
        // Boundary: exactly 0.90 → High
        assert_eq!(DcTier::from_fraction(0.90), DcTier::High);
    }

    // ── SafeFailureFraction ─────────────────────────────────────────────────

    #[test]
    fn sff_valid_range() {
        assert!(SafeFailureFraction::new(0.0).is_ok());
        assert!(SafeFailureFraction::new(0.6).is_ok());
        assert!(SafeFailureFraction::new(1.0).is_ok());
    }

    #[test]
    fn sff_invalid_range() {
        assert!(SafeFailureFraction::new(-0.01).is_err());
        assert!(SafeFailureFraction::new(1.01).is_err());
    }

    // ── ReliabilityBlock ────────────────────────────────────────────────────

    #[test]
    fn reliability_block_pfd_calculation() {
        // λ = 1e-6/h, SFF = 0.9, DC = 0.9, T_proof = 8760 h
        // λ_d = 1e-6 * 0.1 = 1e-7/h
        // λ_du = 1e-7 * 0.1 = 1e-8/h
        // PFD = 1e-8 * 8760 / 2 = 4.38e-5
        let block = ReliabilityBlock::new(1e-6, 0.9, 0.9).unwrap();
        let pfd = block.pfd_avg(8760.0);
        assert!((pfd - 4.38e-5).abs() < 1e-8, "pfd={pfd:.4e}");
    }

    #[test]
    fn reliability_block_pfh() {
        let block = ReliabilityBlock::new(1e-5, 0.8, 0.95).unwrap();
        // λ_du = 1e-5 * 0.2 * 0.05 = 1e-7
        let expected = 1e-5 * 0.2 * 0.05;
        assert!((block.pfh() - expected).abs() < 1e-12);
    }

    #[test]
    fn reliability_block_invalid_inputs() {
        assert!(ReliabilityBlock::new(-1e-6, 0.5, 0.9).is_err());
        assert!(ReliabilityBlock::new(1e-6, 1.5, 0.9).is_err());
        assert!(ReliabilityBlock::new(1e-6, 0.5, -0.1).is_err());
    }

    // ── FaultTree — basic event ─────────────────────────────────────────────

    #[test]
    fn basic_event_pfd() {
        // λ = 1e-6/h, T = 8760 h → PFD = 1e-6 * 8760 / 2 = 4.38e-3
        let mut tree = FaultTree::new();
        let root = tree.add_basic_event(1e-6, 0.0).unwrap();
        let pfd = tree.compute_pfd(root, 8760.0).unwrap();
        assert!((pfd - 4.38e-3).abs() < 1e-7, "pfd={pfd:.4e}");
    }

    #[test]
    fn basic_event_with_beta_reduces_pfd() {
        // beta = 0.1: effective λ = 1e-6 * 0.9
        let mut tree_no = FaultTree::new();
        let r_no = tree_no.add_basic_event(1e-6, 0.0).unwrap();
        let pfd_no = tree_no.compute_pfd(r_no, 8760.0).unwrap();

        let mut tree_b = FaultTree::new();
        let r_b = tree_b.add_basic_event(1e-6, 0.1).unwrap();
        let pfd_beta = tree_b.compute_pfd(r_b, 8760.0).unwrap();

        assert!(pfd_beta < pfd_no);
        let expected = 1e-6 * 0.9 * 8760.0 * 0.5;
        assert!((pfd_beta - expected).abs() < 1e-9);
    }

    // ── FaultTree — AND gate ────────────────────────────────────────────────

    #[test]
    fn and_gate_pfd_is_product() {
        // A: λ=1e-4 → PFD=0.05; B: λ=2e-4 → PFD=0.10 (T=1000)
        let mut tree = FaultTree::new();
        let a = tree.add_basic_event(1e-4, 0.0).unwrap();
        let b = tree.add_basic_event(2e-4, 0.0).unwrap();
        let gate = tree.add_and(&[a, b]).unwrap();
        let pfd = tree.compute_pfd(gate, 1000.0).unwrap();
        let expected = 0.05 * 0.10;
        assert!((pfd - expected).abs() < 1e-10, "and_pfd={pfd:.4e}");
    }

    // ── FaultTree — OR gate ─────────────────────────────────────────────────

    #[test]
    fn or_gate_pfd_complement_product() {
        // PFD_A = 0.05, PFD_B = 0.10 (T=1000)
        // OR: 1−(1−0.05)*(1−0.10) = 0.145
        let mut tree = FaultTree::new();
        let a = tree.add_basic_event(1e-4, 0.0).unwrap();
        let b = tree.add_basic_event(2e-4, 0.0).unwrap();
        let gate = tree.add_or(&[a, b]).unwrap();
        let pfd = tree.compute_pfd(gate, 1000.0).unwrap();
        let expected = 1.0 - (1.0 - 0.05) * (1.0 - 0.10);
        assert!((pfd - expected).abs() < 1e-10, "or_pfd={pfd:.4e}");
    }

    #[test]
    fn nested_and_or_tree() {
        // Top = OR( AND(A,B), C )
        //   A: λ=1e-4, T=1000 → PFD=0.05
        //   B: λ=2e-4, T=1000 → PFD=0.10
        //   C: λ=5e-5, T=1000 → PFD=0.025
        // AND(A,B) PFD = 0.005
        // OR(AND(A,B), C) = 1−(1−0.005)*(1−0.025)
        let mut tree = FaultTree::new();
        let a = tree.add_basic_event(1e-4, 0.0).unwrap();
        let b = tree.add_basic_event(2e-4, 0.0).unwrap();
        let c = tree.add_basic_event(5e-5, 0.0).unwrap();
        let and_ab = tree.add_and(&[a, b]).unwrap();
        let top = tree.add_or(&[and_ab, c]).unwrap();
        let pfd = tree.compute_pfd(top, 1000.0).unwrap();
        let expected = 1.0 - (1.0 - 0.05 * 0.10) * (1.0 - 0.025);
        assert!((pfd - expected).abs() < 1e-9, "nested_pfd={pfd:.6e}");
    }

    #[test]
    fn fault_tree_invalid_interval() {
        let mut tree = FaultTree::new();
        let root = tree.add_basic_event(1e-6, 0.0).unwrap();
        assert!(matches!(
            tree.compute_pfd(root, -1.0),
            Err(DiagnosticError::InvalidTestInterval { .. })
        ));
        assert!(matches!(
            tree.compute_pfd(root, f64::NAN),
            Err(DiagnosticError::InvalidTestInterval { .. })
        ));
    }

    #[test]
    fn fault_tree_too_many_children() {
        let mut tree = FaultTree::new();
        let indices: [u8; 9] = [0; 9];
        assert!(matches!(
            tree.add_and(&indices),
            Err(DiagnosticError::TooManyChildren { .. })
        ));
    }

    // ── DiagnosticMonitor ───────────────────────────────────────────────────

    #[test]
    fn monitor_dc_calculation() {
        let mut mon = DiagnosticMonitor::new();
        mon.record_detected_failure(9e-7).unwrap();
        mon.record_undetected_failure(1e-7).unwrap();
        // DC = 9e-7 / (9e-7 + 1e-7) = 0.9
        let dc = mon.diagnostic_coverage();
        assert!((dc - 0.9).abs() < 1e-10, "DC={dc}");
    }

    #[test]
    fn monitor_sff_calculation() {
        let mut mon = DiagnosticMonitor::new();
        mon.record_safe_failure(5e-7).unwrap();
        mon.record_detected_failure(4e-7).unwrap();
        mon.record_undetected_failure(1e-7).unwrap();
        // SFF = (5e-7 + 4e-7) / (5e-7 + 4e-7 + 1e-7) = 9e-7 / 1e-6 = 0.9
        let sff = mon.safe_failure_fraction();
        assert!((sff - 0.9).abs() < 1e-10, "SFF={sff}");
    }

    #[test]
    fn monitor_dc_zero_when_no_dangerous_failures() {
        let mon = DiagnosticMonitor::new();
        assert_eq!(mon.diagnostic_coverage(), 0.0);
    }

    #[test]
    fn monitor_sff_one_when_no_failures() {
        let mon = DiagnosticMonitor::new();
        assert_eq!(mon.safe_failure_fraction(), 1.0);
    }

    #[test]
    fn monitor_rejects_negative_rate() {
        let mut mon = DiagnosticMonitor::new();
        assert!(matches!(
            mon.record_detected_failure(-1e-6),
            Err(DiagnosticError::InvalidFailureRate { .. })
        ));
    }

    #[test]
    fn monitor_reset_clears_all() {
        let mut mon = DiagnosticMonitor::new();
        mon.record_detected_failure(1e-6).unwrap();
        mon.record_undetected_failure(1e-7).unwrap();
        mon.reset();
        assert_eq!(mon.lambda_total(), 0.0);
        assert_eq!(mon.diagnostic_coverage(), 0.0);
    }
}
