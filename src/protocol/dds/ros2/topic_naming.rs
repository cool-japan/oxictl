//! ROS2 topic/type name encoding: maps ROS2 names to DDS topic/type names.

use heapless::String;

use super::error::Ros2Error;

/// Indicates the kind of ROS2 communication primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ros2TopicKind {
    /// A regular ROS2 topic (publish/subscribe). DDS prefix: `rt/`.
    Topic,
    /// The request half of a ROS2 service. DDS prefix: `rq/`, suffix: `Request`.
    ServiceRequest,
    /// The reply half of a ROS2 service. DDS prefix: `rr/`, suffix: `Reply`.
    ServiceReply,
    /// Service status (rare). DDS prefix: `rs/`, suffix: `Reply`.
    ServiceStatus,
}

/// Whether the ROS2 type belongs to `msg`, `srv`, or `action` sub-namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeNamespace {
    Msg,
    Srv,
    Action,
}

/// Optional suffix appended to the mangled type name before the trailing `_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSuffix {
    Plain,
    Request,
    Response,
}

/// Which action-level topic to address: feedback or status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSubtopic {
    /// Periodic feedback topic: `rt/<action>/_action/feedback`
    Feedback,
    /// Goal status array topic: `rt/<action>/_action/status`
    Status,
}

/// Encode an action feedback or status topic name.
///
/// Example: `encode_action_subtopic(&mut buf, "fibonacci", ActionSubtopic::Feedback)`
/// → `"rt/fibonacci/_action/feedback"`
///
/// Action service sub-topics (`send_goal`, `cancel_goal`, `get_result`) are handled
/// by composing `"<action>/_action/send_goal"` with `encode_topic_name(..., ServiceRequest)`.
pub fn encode_action_subtopic(
    out: &mut String<256>,
    action_name: &str,
    sub: ActionSubtopic,
) -> Result<(), Ros2Error> {
    let name = action_name.strip_prefix('/').unwrap_or(action_name);
    validate_name(name, Ros2Error::InvalidTopicName)?;
    out.clear();
    push_str(out, "rt/")?;
    push_str(out, name)?;
    push_str(out, "/_action/")?;
    let sub_str = match sub {
        ActionSubtopic::Feedback => "feedback",
        ActionSubtopic::Status => "status",
    };
    push_str(out, sub_str)?;
    Ok(())
}

/// Validate that a name component is non-empty, has no whitespace, and no NUL bytes.
fn validate_name(name: &str, err: Ros2Error) -> Result<(), Ros2Error> {
    if name.is_empty() {
        return Err(err);
    }
    for ch in name.chars() {
        if ch == '\0' || ch.is_whitespace() {
            return Err(err);
        }
    }
    Ok(())
}

/// Push a `&str` into a `heapless::String<256>`, returning `NameBufferTooSmall` on overflow.
fn push_str(out: &mut String<256>, s: &str) -> Result<(), Ros2Error> {
    out.push_str(s).map_err(|_| Ros2Error::NameBufferTooSmall)
}

/// Encode a ROS2 topic/service name into its DDS topic name.
///
/// The `ros_name` may or may not have a leading `/`; it is stripped if present.
/// Returns `Ros2Error::InvalidTopicName` if the name is empty, contains whitespace, or NUL.
/// Returns `Ros2Error::NameBufferTooSmall` if the result exceeds 256 bytes.
pub fn encode_topic_name(
    out: &mut String<256>,
    ros_name: &str,
    kind: Ros2TopicKind,
) -> Result<(), Ros2Error> {
    // Strip leading slash
    let name = ros_name.strip_prefix('/').unwrap_or(ros_name);

    validate_name(name, Ros2Error::InvalidTopicName)?;

    out.clear();

    let (prefix, suffix) = match kind {
        Ros2TopicKind::Topic => ("rt/", ""),
        Ros2TopicKind::ServiceRequest => ("rq/", "Request"),
        Ros2TopicKind::ServiceReply => ("rr/", "Reply"),
        Ros2TopicKind::ServiceStatus => ("rs/", "Reply"),
    };

    push_str(out, prefix)?;
    push_str(out, name)?;
    push_str(out, suffix)?;

    Ok(())
}

/// Decode a DDS topic name back to `(Ros2TopicKind, rest_after_prefix)`.
///
/// Returns `None` if the DDS name does not start with a known ROS2 prefix.
/// The returned `&str` is the slice after the prefix (not stripped of suffix).
pub fn decode_topic_name(dds_name: &str) -> Option<(Ros2TopicKind, &str)> {
    if let Some(rest) = dds_name.strip_prefix("rt/") {
        return Some((Ros2TopicKind::Topic, rest));
    }
    if let Some(rest) = dds_name.strip_prefix("rq/") {
        return Some((Ros2TopicKind::ServiceRequest, rest));
    }
    if let Some(rest) = dds_name.strip_prefix("rr/") {
        return Some((Ros2TopicKind::ServiceReply, rest));
    }
    if let Some(rest) = dds_name.strip_prefix("rs/") {
        return Some((Ros2TopicKind::ServiceStatus, rest));
    }
    None
}

/// Encode a ROS2 package + type name into the DDS C++ mangled form.
///
/// Example: `encode_type_name(&mut buf, "rcl_interfaces", TypeNamespace::Msg, "Log", TypeSuffix::Plain)`
/// → `rcl_interfaces::msg::dds_::Log_`
pub fn encode_type_name(
    out: &mut String<256>,
    pkg: &str,
    ns: TypeNamespace,
    type_name: &str,
    suffix: TypeSuffix,
) -> Result<(), Ros2Error> {
    validate_name(pkg, Ros2Error::InvalidTypeName)?;
    validate_name(type_name, Ros2Error::InvalidTypeName)?;

    let ns_str = match ns {
        TypeNamespace::Msg => "msg",
        TypeNamespace::Srv => "srv",
        TypeNamespace::Action => "action",
    };

    let suffix_str = match suffix {
        TypeSuffix::Plain => "",
        TypeSuffix::Request => "_Request",
        TypeSuffix::Response => "_Response",
    };

    out.clear();
    push_str(out, pkg)?;
    push_str(out, "::")?;
    push_str(out, ns_str)?;
    push_str(out, "::dds_::")?;
    push_str(out, type_name)?;
    push_str(out, suffix_str)?;
    push_str(out, "_")?;

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_topic_strips_leading_slash() {
        let mut buf = String::new();
        encode_topic_name(&mut buf, "/chatter", Ros2TopicKind::Topic).unwrap();
        assert_eq!(buf.as_str(), "rt/chatter");
    }

    #[test]
    fn encode_topic_no_leading_slash() {
        let mut buf = String::new();
        encode_topic_name(&mut buf, "chatter", Ros2TopicKind::Topic).unwrap();
        assert_eq!(buf.as_str(), "rt/chatter");
    }

    #[test]
    fn encode_service_request_appends_request() {
        let mut buf = String::new();
        encode_topic_name(&mut buf, "add_two_ints", Ros2TopicKind::ServiceRequest).unwrap();
        assert_eq!(buf.as_str(), "rq/add_two_intsRequest");
    }

    #[test]
    fn encode_service_reply_appends_reply() {
        let mut buf = String::new();
        encode_topic_name(&mut buf, "add_two_ints", Ros2TopicKind::ServiceReply).unwrap();
        assert_eq!(buf.as_str(), "rr/add_two_intsReply");
    }

    #[test]
    fn encode_type_name_msg() {
        let mut buf = String::new();
        encode_type_name(
            &mut buf,
            "std_msgs",
            TypeNamespace::Msg,
            "String",
            TypeSuffix::Plain,
        )
        .unwrap();
        assert_eq!(buf.as_str(), "std_msgs::msg::dds_::String_");
    }

    #[test]
    fn encode_topic_rejects_whitespace() {
        let mut buf = String::new();
        let result = encode_topic_name(&mut buf, "my topic", Ros2TopicKind::Topic);
        assert_eq!(result, Err(Ros2Error::InvalidTopicName));
    }

    #[test]
    fn decode_topic_name_rt_prefix() {
        let result = decode_topic_name("rt/chatter");
        assert_eq!(result, Some((Ros2TopicKind::Topic, "chatter")));
    }

    #[test]
    fn decode_topic_name_rq_prefix() {
        let result = decode_topic_name("rq/add_two_intsRequest");
        assert_eq!(
            result,
            Some((Ros2TopicKind::ServiceRequest, "add_two_intsRequest"))
        );
    }

    #[test]
    fn decode_topic_name_rr_prefix() {
        let result = decode_topic_name("rr/add_two_intsReply");
        assert_eq!(
            result,
            Some((Ros2TopicKind::ServiceReply, "add_two_intsReply"))
        );
    }

    #[test]
    fn decode_topic_name_unknown_prefix_returns_none() {
        let result = decode_topic_name("chatter");
        assert_eq!(result, None);
    }

    #[test]
    fn encode_type_name_srv_request() {
        let mut buf = String::new();
        encode_type_name(
            &mut buf,
            "rcl_interfaces",
            TypeNamespace::Srv,
            "GetParameters",
            TypeSuffix::Request,
        )
        .unwrap();
        assert_eq!(
            buf.as_str(),
            "rcl_interfaces::srv::dds_::GetParameters_Request_"
        );
    }

    #[test]
    fn encode_topic_rejects_empty() {
        let mut buf = String::new();
        let result = encode_topic_name(&mut buf, "", Ros2TopicKind::Topic);
        assert_eq!(result, Err(Ros2Error::InvalidTopicName));
    }

    #[test]
    fn encode_type_name_rejects_empty_pkg() {
        let mut buf = String::new();
        let result = encode_type_name(&mut buf, "", TypeNamespace::Msg, "Log", TypeSuffix::Plain);
        assert_eq!(result, Err(Ros2Error::InvalidTypeName));
    }

    #[test]
    fn encode_type_name_srv_response() {
        let mut buf = String::new();
        encode_type_name(
            &mut buf,
            "rcl_interfaces",
            TypeNamespace::Srv,
            "GetParameters",
            TypeSuffix::Response,
        )
        .unwrap();
        assert_eq!(
            buf.as_str(),
            "rcl_interfaces::srv::dds_::GetParameters_Response_"
        );
    }

    #[test]
    fn encode_action_type_name() {
        let mut buf = String::new();
        encode_type_name(
            &mut buf,
            "example_interfaces",
            TypeNamespace::Action,
            "Fibonacci_SendGoal",
            TypeSuffix::Request,
        )
        .unwrap();
        assert_eq!(
            buf.as_str(),
            "example_interfaces::action::dds_::Fibonacci_SendGoal_Request_"
        );
    }

    #[test]
    fn encode_action_subtopic_feedback() {
        let mut buf = String::new();
        encode_action_subtopic(&mut buf, "fibonacci", ActionSubtopic::Feedback).unwrap();
        assert_eq!(buf.as_str(), "rt/fibonacci/_action/feedback");
    }

    #[test]
    fn encode_action_subtopic_status() {
        let mut buf = String::new();
        encode_action_subtopic(&mut buf, "/fibonacci", ActionSubtopic::Status).unwrap();
        assert_eq!(buf.as_str(), "rt/fibonacci/_action/status");
    }

    #[test]
    fn encode_action_service_topic_send_goal() {
        // Action services use encode_topic_name with composed inner name
        let mut buf = String::new();
        encode_topic_name(
            &mut buf,
            "fibonacci/_action/send_goal",
            Ros2TopicKind::ServiceRequest,
        )
        .unwrap();
        assert_eq!(buf.as_str(), "rq/fibonacci/_action/send_goalRequest");
    }
}
