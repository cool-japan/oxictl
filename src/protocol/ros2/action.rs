//! ROS2 action server/client stub.
//!
//! ROS2 actions are long-running tasks with goal, feedback, and result.
//! This module implements a minimal finite-state action server that can be
//! embedded in a control loop.
//!
//! Mirrors the ROS2 action interface:
//!   - Goal: sent by client to initiate action
//!   - Feedback: periodic progress updates from server
//!   - Result: final outcome when action completes

/// Action server state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    /// No active goal.
    Idle,
    /// Goal received, processing started.
    Executing,
    /// Cancellation requested.
    Canceling,
    /// Goal succeeded.
    Succeeded,
    /// Goal was canceled.
    Canceled,
    /// Goal aborted due to error.
    Aborted,
}

impl ActionStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Canceled | Self::Aborted)
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Executing | Self::Canceling)
    }
}

/// Action server for a generic goal type G, feedback type F, result type R.
pub struct ActionServer<G: Copy, F: Copy, R: Copy> {
    pub status: ActionStatus,
    pub goal: Option<G>,
    pub last_feedback: Option<F>,
    pub result: Option<R>,
    /// Sequence counter (monotonic, for feedback IDs).
    pub seq: u32,
}

impl<G: Copy, F: Copy, R: Copy> ActionServer<G, F, R> {
    pub fn new() -> Self {
        Self {
            status: ActionStatus::Idle,
            goal: None,
            last_feedback: None,
            result: None,
            seq: 0,
        }
    }

    /// Accept a new goal. Returns false if already executing.
    pub fn accept_goal(&mut self, goal: G) -> bool {
        if self.status.is_active() {
            return false;
        }
        self.goal = Some(goal);
        self.last_feedback = None;
        self.result = None;
        self.status = ActionStatus::Executing;
        self.seq += 1;
        true
    }

    /// Publish feedback during execution.
    pub fn publish_feedback(&mut self, feedback: F) {
        if self.status == ActionStatus::Executing || self.status == ActionStatus::Canceling {
            self.last_feedback = Some(feedback);
            self.seq += 1;
        }
    }

    /// Complete goal successfully.
    pub fn succeed(&mut self, result: R) {
        if self.status.is_active() {
            self.result = Some(result);
            self.status = ActionStatus::Succeeded;
        }
    }

    /// Abort goal with an error.
    pub fn abort(&mut self, result: R) {
        if self.status.is_active() {
            self.result = Some(result);
            self.status = ActionStatus::Aborted;
        }
    }

    /// Request cancellation (external).
    pub fn request_cancel(&mut self) -> bool {
        if self.status == ActionStatus::Executing {
            self.status = ActionStatus::Canceling;
            true
        } else {
            false
        }
    }

    /// Complete cancellation.
    pub fn cancel(&mut self, result: R) {
        if self.status == ActionStatus::Canceling {
            self.result = Some(result);
            self.status = ActionStatus::Canceled;
        }
    }

    /// Reset to idle.
    pub fn reset(&mut self) {
        self.status = ActionStatus::Idle;
        self.goal = None;
        self.last_feedback = None;
        self.result = None;
    }
}

impl<G: Copy, F: Copy, R: Copy> Default for ActionServer<G, F, R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Move-to-joint-position action types.
pub mod move_joint {
    /// Goal: target joint positions.
    #[derive(Debug, Clone, Copy)]
    pub struct Goal<const N: usize> {
        pub target: [f64; N],
        pub max_velocity: f64,
    }

    /// Feedback: current positions + error.
    #[derive(Debug, Clone, Copy)]
    pub struct Feedback<const N: usize> {
        pub current: [f64; N],
        pub error_norm: f64,
    }

    /// Result: final error, success flag.
    #[derive(Debug, Clone, Copy)]
    pub struct Result {
        pub final_error: f64,
        pub success: bool,
    }
}

#[cfg(test)]
mod tests {
    use super::move_joint::{Feedback, Goal, Result};
    use super::*;

    type MoveServer = ActionServer<Goal<6>, Feedback<6>, Result>;

    #[test]
    fn action_lifecycle() {
        let mut server = MoveServer::new();
        assert_eq!(server.status, ActionStatus::Idle);

        let goal = Goal {
            target: [0.0; 6],
            max_velocity: 1.0,
        };
        assert!(server.accept_goal(goal));
        assert_eq!(server.status, ActionStatus::Executing);

        server.publish_feedback(Feedback {
            current: [0.1; 6],
            error_norm: 0.1,
        });
        assert!(server.last_feedback.is_some());

        server.succeed(Result {
            final_error: 0.001,
            success: true,
        });
        assert_eq!(server.status, ActionStatus::Succeeded);
        assert!(server.result.unwrap().success);
    }

    #[test]
    fn action_cancel() {
        let mut server = MoveServer::new();
        server.accept_goal(Goal {
            target: [0.0; 6],
            max_velocity: 1.0,
        });
        assert!(server.request_cancel());
        assert_eq!(server.status, ActionStatus::Canceling);
        server.cancel(Result {
            final_error: 0.5,
            success: false,
        });
        assert_eq!(server.status, ActionStatus::Canceled);
    }

    #[test]
    fn action_reject_double_goal() {
        let mut server = MoveServer::new();
        server.accept_goal(Goal {
            target: [0.0; 6],
            max_velocity: 1.0,
        });
        assert!(!server.accept_goal(Goal {
            target: [1.0; 6],
            max_velocity: 0.5
        }));
    }

    #[test]
    fn action_abort() {
        let mut server = MoveServer::new();
        server.accept_goal(Goal {
            target: [0.0; 6],
            max_velocity: 1.0,
        });
        server.abort(Result {
            final_error: 999.0,
            success: false,
        });
        assert_eq!(server.status, ActionStatus::Aborted);
    }
}
