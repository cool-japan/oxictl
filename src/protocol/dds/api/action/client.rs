//! `ActionClient<A>` — client-side interface for a ROS2 action.

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::api::service::ServiceClient;
use crate::protocol::dds::api::subscription::Subscription;
use crate::protocol::dds::ros2::msgs::action_msgs::{
    CancelGoal, CancelGoalRequest, GoalInfo, GoalStatusArray,
};
use crate::protocol::dds::ros2::msgs::builtin_interfaces::Time;
use crate::protocol::dds::ros2::msgs::unique_identifier_msgs::Uuid;

use super::types::{Action, GetResultRequest, GetResultService, SendGoalRequest, SendGoalService};

/// Client-side handle for a ROS2 action.
///
/// Created by [`super::create_client`].  Holds the two service clients
/// (`SendGoal`, `GetResult`), the cancel service client, and subscriptions
/// for feedback and goal-status topics.
pub struct ActionClient<A: Action> {
    send_goal_client: ServiceClient<SendGoalService<A>>,
    get_result_client: ServiceClient<GetResultService<A>>,
    cancel_client: ServiceClient<CancelGoal>,
    feedback_sub: Subscription<A::Feedback>,
    status_sub: Subscription<GoalStatusArray>,
}

impl<A: Action> ActionClient<A> {
    /// Construct an `ActionClient` from pre-created service clients and
    /// subscriptions.  Prefer [`super::create_client`] over calling this
    /// directly.
    pub(crate) fn new(
        send_goal_client: ServiceClient<SendGoalService<A>>,
        get_result_client: ServiceClient<GetResultService<A>>,
        cancel_client: ServiceClient<CancelGoal>,
        feedback_sub: Subscription<A::Feedback>,
        status_sub: Subscription<GoalStatusArray>,
    ) -> Self {
        Self {
            send_goal_client,
            get_result_client,
            cancel_client,
            feedback_sub,
            status_sub,
        }
    }

    /// Send a goal to the action server.
    ///
    /// Returns the sequence number assigned to this request, which can be
    /// matched against the response returned by [`take_goal_responses`].
    pub fn send_goal(
        &mut self,
        participant: &mut Participant,
        goal_id: Uuid,
        goal: A::Goal,
    ) -> Result<i64, DdsApiError> {
        let req = SendGoalRequest::<A> { goal_id, goal };
        self.send_goal_client.send_request(participant, &req)
    }

    /// Drain buffered `SendGoal` responses.
    ///
    /// Returns `(sequence_number, accepted, stamp)` tuples.
    pub fn take_goal_responses(&mut self, participant: &mut Participant) -> Vec<(i64, bool, Time)> {
        self.send_goal_client
            .take_responses(participant)
            .into_iter()
            .map(|(seq, resp)| (seq, resp.accepted, resp.stamp))
            .collect()
    }

    /// Request the result of a previously accepted goal.
    ///
    /// Returns the sequence number assigned to this request.
    pub fn request_result(
        &mut self,
        participant: &mut Participant,
        goal_id: Uuid,
    ) -> Result<i64, DdsApiError> {
        let req = GetResultRequest { goal_id };
        self.get_result_client.send_request(participant, &req)
    }

    /// Drain buffered `GetResult` responses.
    ///
    /// Returns `(sequence_number, status, result)` tuples.
    pub fn take_results(&mut self, participant: &mut Participant) -> Vec<(i64, i8, A::Result)> {
        self.get_result_client
            .take_responses(participant)
            .into_iter()
            .map(|(seq, resp)| (seq, resp.status, resp.result))
            .collect()
    }

    /// Request cancellation of a goal identified by `goal_id`.
    ///
    /// Returns the sequence number assigned to the cancel request.
    pub fn cancel_goal(
        &mut self,
        participant: &mut Participant,
        goal_id: Uuid,
    ) -> Result<i64, DdsApiError> {
        let req = CancelGoalRequest {
            goal_info: GoalInfo {
                goal_id,
                stamp: Time::default(),
            },
        };
        self.cancel_client.send_request(participant, &req)
    }

    /// Drain buffered feedback messages from the action server.
    pub fn take_feedback(&mut self, participant: &mut Participant) -> Vec<A::Feedback> {
        participant
            .take(&self.feedback_sub)
            .into_iter()
            .map(|s| s.data)
            .collect()
    }

    /// Drain buffered goal-status arrays published by the action server.
    pub fn take_status(&mut self, participant: &mut Participant) -> Vec<GoalStatusArray> {
        participant
            .take(&self.status_sub)
            .into_iter()
            .map(|s| s.data)
            .collect()
    }
}
