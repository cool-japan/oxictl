//! `ActionServer<A>` — server-side implementation of a ROS2 action.

use heapless::Vec as HVec;

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::api::publisher::Publisher;
use crate::protocol::dds::api::service::wrappers::{ReplyWrapper, RequestWrapper};
use crate::protocol::dds::api::subscription::Subscription;
use crate::protocol::dds::ros2::msgs::action_msgs::{
    goal_status, CancelGoal, CancelGoalResponse, GoalInfo, GoalStatusArray,
};
use crate::protocol::dds::ros2::msgs::builtin_interfaces::Time;

use super::types::{
    Action, ActionHandler, ExecuteResult, GetResultResponse, GetResultService, GoalOutcome,
    GoalState, SendGoalResponse, SendGoalService,
};

/// All pub/sub handles needed to construct an [`ActionServer`].
///
/// Grouping them in a struct avoids the too-many-arguments clippy warning
/// and makes the construction site self-documenting.
pub(crate) struct ActionServerParts<A: Action> {
    /// Subscription for incoming `SendGoal` requests.
    pub send_goal_req_sub: Subscription<RequestWrapper<SendGoalService<A>>>,
    /// Publisher for `SendGoal` replies.
    pub send_goal_rep_pub: Publisher<ReplyWrapper<SendGoalService<A>>>,
    /// Subscription for incoming `CancelGoal` requests.
    pub cancel_goal_req_sub: Subscription<RequestWrapper<CancelGoal>>,
    /// Publisher for `CancelGoal` replies.
    pub cancel_goal_rep_pub: Publisher<ReplyWrapper<CancelGoal>>,
    /// Subscription for incoming `GetResult` requests.
    pub get_result_req_sub: Subscription<RequestWrapper<GetResultService<A>>>,
    /// Publisher for `GetResult` replies.
    pub get_result_rep_pub: Publisher<ReplyWrapper<GetResultService<A>>>,
    /// Publisher for intermediate feedback messages.
    pub feedback_pub: Publisher<A::Feedback>,
    /// Publisher for the `GoalStatusArray` topic.
    pub status_pub: Publisher<GoalStatusArray>,
}

/// Server-side implementation of a ROS2 action.
pub struct ActionServer<A: Action> {
    send_goal_req_sub: Subscription<RequestWrapper<SendGoalService<A>>>,
    send_goal_rep_pub: Publisher<ReplyWrapper<SendGoalService<A>>>,
    cancel_goal_req_sub: Subscription<RequestWrapper<CancelGoal>>,
    cancel_goal_rep_pub: Publisher<ReplyWrapper<CancelGoal>>,
    get_result_req_sub: Subscription<RequestWrapper<GetResultService<A>>>,
    get_result_rep_pub: Publisher<ReplyWrapper<GetResultService<A>>>,
    feedback_pub: Publisher<A::Feedback>,
    status_pub: Publisher<GoalStatusArray>,
    goals: HVec<GoalState<A>, 16>,
}

impl<A: Action> ActionServer<A> {
    /// Create a new `ActionServer` from pre-created pub/sub handles.
    pub(crate) fn new(parts: ActionServerParts<A>) -> Self {
        Self {
            send_goal_req_sub: parts.send_goal_req_sub,
            send_goal_rep_pub: parts.send_goal_rep_pub,
            cancel_goal_req_sub: parts.cancel_goal_req_sub,
            cancel_goal_rep_pub: parts.cancel_goal_rep_pub,
            get_result_req_sub: parts.get_result_req_sub,
            get_result_rep_pub: parts.get_result_rep_pub,
            feedback_pub: parts.feedback_pub,
            status_pub: parts.status_pub,
            goals: HVec::new(),
        }
    }

    /// Drive one spin cycle: accept/reject goals, process cancellations, execute
    /// ready goals, drain feedback, answer result queries, and publish status.
    pub fn process<H: ActionHandler<A>>(
        &mut self,
        participant: &mut Participant,
        handler: &mut H,
    ) -> Result<(), DdsApiError> {
        // Step 1: SendGoal requests
        let send_goal_samples = participant.take(&self.send_goal_req_sub);
        for sample in send_goal_samples {
            let RequestWrapper { header, body: req } = sample.data;
            let outcome = handler.accept_goal(&req.goal_id, &req.goal);
            let accepted = matches!(outcome, GoalOutcome::Accept);
            if accepted && !self.goals.is_full() {
                let gs = GoalState::<A>::new(req.goal_id, req.goal);
                let _ = self.goals.push(gs);
            }
            let response = SendGoalResponse {
                accepted,
                stamp: Time::default(),
            };
            let reply = ReplyWrapper::<SendGoalService<A>> {
                header,
                body: response,
            };
            participant.publish(&self.send_goal_rep_pub, &reply)?;
        }

        // Step 2: CancelGoal requests
        let cancel_samples = participant.take(&self.cancel_goal_req_sub);
        for sample in cancel_samples {
            let RequestWrapper { header, body: req } = sample.data;
            let target_id = req.goal_info.goal_id;
            let mut goals_canceling: HVec<GoalInfo, 16> = HVec::new();
            for gs in self.goals.iter_mut() {
                if gs.goal_info.goal_id == target_id
                    && (gs.status == goal_status::ACCEPTED || gs.status == goal_status::EXECUTING)
                    && handler.cancel_goal(&gs.goal_info.goal_id)
                {
                    gs.status = goal_status::CANCELING;
                    let _ = goals_canceling.push(gs.goal_info.clone());
                }
            }
            let response = CancelGoalResponse {
                return_code: 0,
                goals_canceling,
            };
            let reply = ReplyWrapper::<CancelGoal> {
                header,
                body: response,
            };
            participant.publish(&self.cancel_goal_rep_pub, &reply)?;
        }

        // Step 3: Execute goals
        for gs in self.goals.iter_mut() {
            if gs.status == goal_status::ACCEPTED {
                gs.status = goal_status::EXECUTING;
            }
            if gs.status == goal_status::EXECUTING || gs.status == goal_status::CANCELING {
                let goal_id = gs.goal_info.goal_id;
                let goal = gs.goal.clone();
                let mut collected_feedback: HVec<A::Feedback, 8> = HVec::new();
                let exec_result = {
                    let mut fb_cb = |fb: A::Feedback| {
                        let _ = collected_feedback.push(fb);
                    };
                    handler.execute_goal(&goal_id, &goal, &mut fb_cb)
                };
                for fb in collected_feedback {
                    let _ = gs.pending_feedback.push(fb);
                }
                match exec_result {
                    ExecuteResult::Succeeded(r) => {
                        gs.status = goal_status::SUCCEEDED;
                        gs.result = Some(r);
                    }
                    ExecuteResult::Canceled(r) => {
                        gs.status = goal_status::CANCELED;
                        gs.result = Some(r);
                    }
                    ExecuteResult::Aborted(r) => {
                        gs.status = goal_status::ABORTED;
                        gs.result = Some(r);
                    }
                }
            }
        }

        // Drain feedback
        for gs in self.goals.iter_mut() {
            while let Some(fb) = gs.pending_feedback.pop() {
                participant.publish(&self.feedback_pub, &fb)?;
            }
        }

        // Step 4: GetResult requests
        let get_result_samples = participant.take(&self.get_result_req_sub);
        for sample in get_result_samples {
            let RequestWrapper { header, body: req } = sample.data;
            let result_opt = self.goals.iter().find_map(|gs| {
                if gs.goal_info.goal_id == req.goal_id
                    && (gs.status == goal_status::SUCCEEDED
                        || gs.status == goal_status::CANCELED
                        || gs.status == goal_status::ABORTED)
                {
                    gs.result.as_ref().map(|r| (gs.status, r.clone()))
                } else {
                    None
                }
            });
            if let Some((status, result)) = result_opt {
                let response = GetResultResponse::<A> { status, result };
                let reply = ReplyWrapper::<GetResultService<A>> {
                    header,
                    body: response,
                };
                participant.publish(&self.get_result_rep_pub, &reply)?;
            }
        }

        // Step 5: Publish GoalStatusArray
        let mut status_array = GoalStatusArray::default();
        for gs in &self.goals {
            let _ = status_array.status_list.push(gs.to_goal_status());
        }
        participant.publish(&self.status_pub, &status_array)?;
        Ok(())
    }
}
