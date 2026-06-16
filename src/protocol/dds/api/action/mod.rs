//! ROS2 action layer — type-safe goal/feedback/result on top of DDS services.

pub mod client;
pub mod server;
pub mod types;

pub use client::ActionClient;
pub use server::ActionServer;
use server::ActionServerParts;
pub use types::{
    Action, ActionHandler, ExecuteResult, GetResultRequest, GetResultResponse, GetResultService,
    GoalOutcome, SendGoalRequest, SendGoalResponse, SendGoalService,
};

use heapless::String as HString;

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::api::service::{
    create_client as create_svc_client, ReplyWrapper, RequestWrapper,
};
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::ros2::msgs::action_msgs::{CancelGoal, GoalStatusArray};
use crate::protocol::dds::ros2::topic_naming::{encode_topic_name, Ros2TopicKind};

fn action_sub_name(action_name: &str, suffix: &str) -> Result<HString<256>, DdsApiError> {
    let mut s: HString<256> = HString::new();
    s.push_str(action_name)
        .map_err(|_| DdsApiError::Serialization("action name too long"))?;
    s.push_str(suffix)
        .map_err(|_| DdsApiError::Serialization("action name suffix too long"))?;
    Ok(s)
}

fn encode_action_topic(topic_name: &str, kind: Ros2TopicKind) -> Result<HString<256>, DdsApiError> {
    let mut encoded: HString<256> = HString::new();
    encode_topic_name(&mut encoded, topic_name, kind).map_err(|_| DdsApiError::TopicNameTooLong)?;
    Ok(encoded)
}

/// Create an [`ActionServer`] for action `action_name`, wiring up all six
/// request/reply service pairs plus the feedback and status publishers.
pub fn create_action_server<A: Action>(
    participant: &mut Participant,
    action_name: &str,
    qos: &QosProfile,
) -> Result<ActionServer<A>, DdsApiError> {
    // SendGoal
    let sg_name = action_sub_name(action_name, "/_action/send_goal")?;
    let sg_req_topic = encode_action_topic(sg_name.as_str(), Ros2TopicKind::ServiceRequest)?;
    let sg_rep_topic = encode_action_topic(sg_name.as_str(), Ros2TopicKind::ServiceReply)?;
    let send_goal_req_sub = participant
        .create_subscription::<RequestWrapper<SendGoalService<A>>>(sg_req_topic.as_str(), qos)?;
    let send_goal_rep_pub = participant
        .create_publisher::<ReplyWrapper<SendGoalService<A>>>(sg_rep_topic.as_str(), qos)?;

    // CancelGoal
    let cg_name = action_sub_name(action_name, "/_action/cancel_goal")?;
    let cg_req_topic = encode_action_topic(cg_name.as_str(), Ros2TopicKind::ServiceRequest)?;
    let cg_rep_topic = encode_action_topic(cg_name.as_str(), Ros2TopicKind::ServiceReply)?;
    let cancel_goal_req_sub = participant
        .create_subscription::<RequestWrapper<CancelGoal>>(cg_req_topic.as_str(), qos)?;
    let cancel_goal_rep_pub =
        participant.create_publisher::<ReplyWrapper<CancelGoal>>(cg_rep_topic.as_str(), qos)?;

    // GetResult
    let gr_name = action_sub_name(action_name, "/_action/get_result")?;
    let gr_req_topic = encode_action_topic(gr_name.as_str(), Ros2TopicKind::ServiceRequest)?;
    let gr_rep_topic = encode_action_topic(gr_name.as_str(), Ros2TopicKind::ServiceReply)?;
    let get_result_req_sub = participant
        .create_subscription::<RequestWrapper<GetResultService<A>>>(gr_req_topic.as_str(), qos)?;
    let get_result_rep_pub = participant
        .create_publisher::<ReplyWrapper<GetResultService<A>>>(gr_rep_topic.as_str(), qos)?;

    // Feedback publisher
    let fb_name = action_sub_name(action_name, "/_action/feedback")?;
    let fb_topic = encode_action_topic(fb_name.as_str(), Ros2TopicKind::Topic)?;
    let feedback_pub = participant.create_publisher::<A::Feedback>(fb_topic.as_str(), qos)?;

    // Status publisher
    let st_name = action_sub_name(action_name, "/_action/status")?;
    let st_topic = encode_action_topic(st_name.as_str(), Ros2TopicKind::Topic)?;
    let status_pub = participant.create_publisher::<GoalStatusArray>(st_topic.as_str(), qos)?;

    Ok(ActionServer::new(ActionServerParts {
        send_goal_req_sub,
        send_goal_rep_pub,
        cancel_goal_req_sub,
        cancel_goal_rep_pub,
        get_result_req_sub,
        get_result_rep_pub,
        feedback_pub,
        status_pub,
    }))
}

/// Create an [`ActionClient`] for action `action_name`, wiring up all three
/// service clients plus the feedback and status subscriptions.
pub fn create_action_client<A: Action>(
    participant: &mut Participant,
    action_name: &str,
    qos: &QosProfile,
) -> Result<ActionClient<A>, DdsApiError> {
    let sg_name = action_sub_name(action_name, "/_action/send_goal")?;
    let send_goal_client =
        create_svc_client::<SendGoalService<A>>(participant, sg_name.as_str(), qos)?;

    let gr_name = action_sub_name(action_name, "/_action/get_result")?;
    let get_result_client =
        create_svc_client::<GetResultService<A>>(participant, gr_name.as_str(), qos)?;

    let cg_name = action_sub_name(action_name, "/_action/cancel_goal")?;
    let cancel_client = create_svc_client::<CancelGoal>(participant, cg_name.as_str(), qos)?;

    // Feedback subscription
    let fb_name = action_sub_name(action_name, "/_action/feedback")?;
    let fb_topic = encode_action_topic(fb_name.as_str(), Ros2TopicKind::Topic)?;
    let feedback_sub = participant.create_subscription::<A::Feedback>(fb_topic.as_str(), qos)?;

    // Status subscription
    let st_name = action_sub_name(action_name, "/_action/status")?;
    let st_topic = encode_action_topic(st_name.as_str(), Ros2TopicKind::Topic)?;
    let status_sub = participant.create_subscription::<GoalStatusArray>(st_topic.as_str(), qos)?;

    Ok(ActionClient::new(
        send_goal_client,
        get_result_client,
        cancel_client,
        feedback_sub,
        status_sub,
    ))
}
