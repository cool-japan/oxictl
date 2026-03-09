/// Networked control systems: event-triggered sampling, self-triggered control,
/// and multi-agent consensus protocols.
///
/// This module provides no-std, allocation-free implementations of:
///
/// - **Event-triggered control** — static (Tabuada 2007) and dynamic event
///   triggers that reduce communication bandwidth by transmitting state
///   information only when a trigger condition is violated.
/// - **Self-triggered control** — precomputes the next required sampling instant
///   from the current state, eliminating continuous monitoring entirely.
/// - **Multi-agent consensus** — leaderless average consensus, leader-following
///   consensus, and distributed gradient descent (ADMM-lite) over fixed graphs.
///
/// All implementations are generic over `S: ControlScalar` and use const-generic
/// arrays for zero heap allocation (`no_std` compatible).
pub mod consensus;
pub mod event_triggered;
pub mod self_triggered;

// ── re-exports ─────────────────────────────────────────────────────────────────

pub use consensus::{
    AgentGraph, AverageConsensus, DistributedGradientDescent, LeaderFollowingConsensus,
};
pub use event_triggered::{
    DynamicTrigger, DynamicTriggerConfig, EventTriggeredController, InterEventTimer, StaticTrigger,
    StaticTriggerConfig,
};
pub use self_triggered::{SelfTrigger, SelfTriggeredLqr};

// ── error type ─────────────────────────────────────────────────────────────────

/// Errors that can arise in networked control computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkedError {
    /// The graph topology is invalid (e.g., non-symmetric weights, self-loops).
    InvalidTopology,

    /// Fewer agents than required for a meaningful consensus computation.
    InsufficientAgents,

    /// A numerical issue (NaN, Inf, division by zero) was detected.
    NumericalError,
}
