//! ROS2 action type system — `Action` trait, associated request/response types,
//! goal-state bookkeeping, and execution outcome types.

use heapless::Vec as HVec;

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::service::wrappers::{Service, ServiceField};
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::ros2::msgs::action_msgs::{goal_status, GoalInfo, GoalStatus};
use crate::protocol::dds::ros2::msgs::builtin_interfaces::Time;
use crate::protocol::dds::ros2::msgs::unique_identifier_msgs::Uuid;

// ─── Action trait ─────────────────────────────────────────────────────────────

/// Describes a complete ROS2 action by associating `Goal`, `Result`, and
/// `Feedback` types together with their DDS type-name strings.
///
/// Implement this trait for a unit struct named after the action, e.g.:
/// ```rust,ignore
/// struct NavigateToPose;
/// impl Action for NavigateToPose {
///     type Goal     = NavigateToPose_Goal;
///     type Result   = NavigateToPose_Result;
///     type Feedback = NavigateToPose_Feedback;
///     const SEND_GOAL_REQUEST_TYPE_NAME:  &'static str = "...";
///     const SEND_GOAL_RESPONSE_TYPE_NAME: &'static str = "...";
///     const GET_RESULT_REQUEST_TYPE_NAME:  &'static str = "...";
///     const GET_RESULT_RESPONSE_TYPE_NAME: &'static str = "...";
///     const FEEDBACK_TYPE_NAME:           &'static str = "...";
/// }
/// ```
pub trait Action: Sized {
    /// CDR body type for the goal payload sent by the action client.
    type Goal: ServiceField + Clone;
    /// CDR body type for the result payload returned by the action server.
    type Result: ServiceField + Clone;
    /// DDS topic type for feedback messages published during execution.
    type Feedback: DdsType + Clone;

    /// DDS type name for the `SendGoal` request wrapper topic.
    const SEND_GOAL_REQUEST_TYPE_NAME: &'static str;
    /// DDS type name for the `SendGoal` response wrapper topic.
    const SEND_GOAL_RESPONSE_TYPE_NAME: &'static str;
    /// DDS type name for the `GetResult` request wrapper topic.
    const GET_RESULT_REQUEST_TYPE_NAME: &'static str;
    /// DDS type name for the `GetResult` response wrapper topic.
    const GET_RESULT_RESPONSE_TYPE_NAME: &'static str;
    /// DDS type name for the feedback topic.
    const FEEDBACK_TYPE_NAME: &'static str;
}

// ─── ActionHandler trait ──────────────────────────────────────────────────────

/// Trait implemented by application code that handles incoming action goals.
///
/// The action server calls these methods in response to client requests.
pub trait ActionHandler<A: Action> {
    /// Decide whether to accept or reject a newly received goal.
    ///
    /// Called synchronously inside `spin_once`; must not block.
    fn accept_goal(&mut self, goal_id: &Uuid, goal: &A::Goal) -> GoalOutcome;

    /// Execute an accepted goal to completion.
    ///
    /// `feedback_cb` may be called zero or more times to publish intermediate
    /// feedback.  Returns the terminal [`ExecuteResult`] when done.
    fn execute_goal(
        &mut self,
        goal_id: &Uuid,
        goal: &A::Goal,
        feedback_cb: &mut dyn FnMut(A::Feedback),
    ) -> ExecuteResult<A>;

    /// Request cancellation of an in-progress goal.
    ///
    /// Returns `true` if the goal was successfully marked for cancellation.
    fn cancel_goal(&mut self, goal_id: &Uuid) -> bool;
}

// ─── GoalOutcome ──────────────────────────────────────────────────────────────

/// Decision returned by [`ActionHandler::accept_goal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalOutcome {
    /// The server accepts the goal and will begin execution.
    Accept,
    /// The server rejects the goal; no execution will occur.
    Reject,
}

// ─── ExecuteResult ────────────────────────────────────────────────────────────

/// Terminal outcome of goal execution, returned by [`ActionHandler::execute_goal`].
pub enum ExecuteResult<A: Action> {
    /// Goal completed successfully; carries the result payload.
    Succeeded(A::Result),
    /// Goal was canceled before or during execution; carries the (partial) result.
    Canceled(A::Result),
    /// Goal execution failed; carries the result payload describing the failure.
    Aborted(A::Result),
}

// ─── SendGoalRequest ──────────────────────────────────────────────────────────

/// CDR body for the `SendGoal` service request: a UUID + the action-specific goal.
pub struct SendGoalRequest<A: Action> {
    /// Client-assigned UUID identifying this goal instance.
    pub goal_id: Uuid,
    /// Action-specific goal payload.
    pub goal: A::Goal,
}

impl<A: Action> Clone for SendGoalRequest<A>
where
    A::Goal: Clone,
{
    fn clone(&self) -> Self {
        Self {
            goal_id: self.goal_id,
            goal: self.goal.clone(),
        }
    }
}

impl<A: Action> ServiceField for SendGoalRequest<A> {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.goal_id.serialize_inner(w)?;
        self.goal.serialize_inner(w)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let goal_id = Uuid::deserialize_inner(r)?;
        let goal = A::Goal::deserialize_inner(r)?;
        Ok(Self { goal_id, goal })
    }
}

// ─── SendGoalResponse ─────────────────────────────────────────────────────────

/// CDR body for the `SendGoal` service response: accepted flag + stamp.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SendGoalResponse {
    /// `true` if the server accepted the goal.
    pub accepted: bool,
    /// Server-side timestamp at which the goal was accepted (zero if rejected).
    pub stamp: Time,
}

impl ServiceField for SendGoalResponse {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_u8(self.accepted as u8)?;
        self.stamp.serialize_inner(w)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let accepted = r.read_u8()? != 0;
        let stamp = Time::deserialize_inner(r)?;
        Ok(Self { accepted, stamp })
    }
}

// ─── SendGoalService ──────────────────────────────────────────────────────────

/// Service descriptor for the `SendGoal` RPC — pairs [`SendGoalRequest`] with
/// [`SendGoalResponse`] and provides the DDS type names from the `Action` trait.
pub struct SendGoalService<A: Action>(core::marker::PhantomData<A>);

impl<A: Action> Service for SendGoalService<A> {
    type Request = SendGoalRequest<A>;
    type Response = SendGoalResponse;
    const REQUEST_TYPE_NAME: &'static str = A::SEND_GOAL_REQUEST_TYPE_NAME;
    const RESPONSE_TYPE_NAME: &'static str = A::SEND_GOAL_RESPONSE_TYPE_NAME;
}

// ─── GetResultRequest ─────────────────────────────────────────────────────────

/// CDR body for the `GetResult` service request: just the goal UUID.
#[derive(Debug, Clone, PartialEq)]
pub struct GetResultRequest {
    /// UUID of the goal whose result is being queried.
    pub goal_id: Uuid,
}

impl ServiceField for GetResultRequest {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.goal_id.serialize_inner(w)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let goal_id = Uuid::deserialize_inner(r)?;
        Ok(Self { goal_id })
    }
}

// ─── GetResultResponse ────────────────────────────────────────────────────────

/// CDR body for the `GetResult` service response: goal status + result payload.
pub struct GetResultResponse<A: Action> {
    /// Final status code; one of the `goal_status::*` constants.
    pub status: i8,
    /// Action-specific result payload.
    pub result: A::Result,
}

impl<A: Action> Clone for GetResultResponse<A>
where
    A::Result: Clone,
{
    fn clone(&self) -> Self {
        Self {
            status: self.status,
            result: self.result.clone(),
        }
    }
}

impl<A: Action> ServiceField for GetResultResponse<A> {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_u8(self.status as u8)?;
        self.result.serialize_inner(w)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let status = r.read_u8()? as i8;
        let result = A::Result::deserialize_inner(r)?;
        Ok(Self { status, result })
    }
}

// ─── GetResultService ─────────────────────────────────────────────────────────

/// Service descriptor for the `GetResult` RPC — pairs [`GetResultRequest`] with
/// [`GetResultResponse`] and provides the DDS type names from the `Action` trait.
pub struct GetResultService<A: Action>(core::marker::PhantomData<A>);

impl<A: Action> Service for GetResultService<A> {
    type Request = GetResultRequest;
    type Response = GetResultResponse<A>;
    const REQUEST_TYPE_NAME: &'static str = A::GET_RESULT_REQUEST_TYPE_NAME;
    const RESPONSE_TYPE_NAME: &'static str = A::GET_RESULT_RESPONSE_TYPE_NAME;
}

// ─── GoalState ────────────────────────────────────────────────────────────────

/// Internal bookkeeping for a single in-flight goal on the action server.
///
/// Tracks the goal payload, current status, optional result, and any feedback
/// messages queued for publication.
pub(crate) struct GoalState<A: Action> {
    /// `GoalInfo` (UUID + accept-time stamp) used in status publications.
    pub goal_info: GoalInfo,
    /// The goal payload received from the client.
    pub goal: A::Goal,
    /// Current goal status; one of the `goal_status::*` constants.
    pub status: i8,
    /// Final result, set once execution reaches a terminal state.
    pub result: Option<A::Result>,
    /// Feedback messages queued for DDS publication (capacity 8).
    pub pending_feedback: HVec<A::Feedback, 8>,
}

impl<A: Action> GoalState<A> {
    /// Construct a new `GoalState` in the `ACCEPTED` status.
    pub(crate) fn new(goal_id: Uuid, goal: A::Goal) -> Self {
        Self {
            goal_info: GoalInfo {
                goal_id,
                stamp: Time::default(),
            },
            goal,
            status: goal_status::ACCEPTED,
            result: None,
            pending_feedback: HVec::new(),
        }
    }

    /// Snapshot the current state as a [`GoalStatus`] for status-array publication.
    pub(crate) fn to_goal_status(&self) -> GoalStatus {
        GoalStatus {
            goal_info: self.goal_info.clone(),
            status: self.status,
        }
    }
}
