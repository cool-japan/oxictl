//! ROS2 builtin endpoint definitions and the standard builtin endpoint set.

use crate::protocol::dds::discovery::qos_profile::QosProfile;

// ─── BuiltinEndpointSet bit flags ────────────────────────────────────────────

/// Bit for `PID_BUILTIN_ENDPOINT_SET`: participant announcer.
pub const DISC_BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER: u32 = 0x0000_0001;
pub const DISC_BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR: u32 = 0x0000_0002;
pub const DISC_BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER: u32 = 0x0000_0004;
pub const DISC_BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR: u32 = 0x0000_0008;
pub const DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER: u32 = 0x0000_0010;
pub const DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR: u32 = 0x0000_0020;
pub const DISC_BUILTIN_ENDPOINT_TOPICS_ANNOUNCER: u32 = 0x0000_0400;
pub const DISC_BUILTIN_ENDPOINT_TOPICS_DETECTOR: u32 = 0x0000_0800;

/// The default builtin endpoint set advertised by a ROS2 participant.
pub const ROS2_DEFAULT_ENDPOINT_SET: u32 = DISC_BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER
    | DISC_BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR
    | DISC_BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER
    | DISC_BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR
    | DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER
    | DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR;

// ─── ROS2 named builtin endpoints ────────────────────────────────────────────

/// A well-known ROS2 builtin endpoint with its fixed topic/type names and QoS.
pub struct Ros2BuiltinEndpoint {
    /// Pre-mangled DDS topic name (e.g. `"rt/rosout"`).
    pub topic_name: &'static str,
    /// Pre-mangled DDS type name (e.g. `"rcl_interfaces::msg::dds_::Log_"`).
    pub type_name: &'static str,
    /// Recommended QoS profile for this endpoint.
    pub qos: QosProfile,
}

/// `/rosout` — subscriber collects log output from all nodes.
/// QoS: Reliable / TransientLocal / KeepLast=1000.
pub const ROS2_BUILTIN_ROSOUT: Ros2BuiltinEndpoint = Ros2BuiltinEndpoint {
    topic_name: "rt/rosout",
    type_name: "rcl_interfaces::msg::dds_::Log_",
    qos: QosProfile::ros2_rosout(),
};

/// `/parameter_events` — parameter change notifications.
/// QoS: Reliable / Volatile / KeepLast=1000.
pub const ROS2_BUILTIN_PARAMETER_EVENTS: Ros2BuiltinEndpoint = Ros2BuiltinEndpoint {
    topic_name: "rt/parameter_events",
    type_name: "rcl_interfaces::msg::dds_::ParameterEvent_",
    qos: QosProfile::ros2_parameter_events(),
};

/// `/clock` — ROS time source.
/// QoS: BestEffort / Volatile / KeepLast=1.
pub const ROS2_BUILTIN_CLOCK: Ros2BuiltinEndpoint = Ros2BuiltinEndpoint {
    topic_name: "rt/clock",
    type_name: "rosgraph_msgs::msg::dds_::Clock_",
    qos: QosProfile::ros2_clock(),
};

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::discovery::qos::{DurabilityKind, HistoryKind, ReliabilityKind};

    #[test]
    fn rosout_uses_reliable_transient_local() {
        let qos = &ROS2_BUILTIN_ROSOUT.qos;
        assert_eq!(qos.reliability.kind, ReliabilityKind::Reliable);
        assert_eq!(qos.durability.kind, DurabilityKind::TransientLocal);
        assert_eq!(qos.history.kind, HistoryKind::KeepLast);
        assert_eq!(qos.history.depth, 1000);
    }

    #[test]
    fn parameter_events_uses_reliable_keep_last_1000() {
        let qos = &ROS2_BUILTIN_PARAMETER_EVENTS.qos;
        assert_eq!(qos.reliability.kind, ReliabilityKind::Reliable);
        assert_eq!(qos.history.kind, HistoryKind::KeepLast);
        assert_eq!(qos.history.depth, 1000);
    }

    #[test]
    fn clock_uses_best_effort_keep_last_1() {
        // ROS2_BUILTIN_CLOCK uses ros2_clock() = BestEffort / KeepLast=1 / Volatile.
        let qos = &ROS2_BUILTIN_CLOCK.qos;
        assert_eq!(qos.reliability.kind, ReliabilityKind::BestEffort);
        assert_eq!(qos.history.kind, HistoryKind::KeepLast);
        assert_eq!(qos.history.depth, 1);
    }

    #[test]
    fn rosout_topic_and_type_names_correct() {
        assert_eq!(ROS2_BUILTIN_ROSOUT.topic_name, "rt/rosout");
        assert_eq!(
            ROS2_BUILTIN_ROSOUT.type_name,
            "rcl_interfaces::msg::dds_::Log_"
        );
    }

    #[test]
    fn parameter_events_topic_and_type_names_correct() {
        assert_eq!(
            ROS2_BUILTIN_PARAMETER_EVENTS.topic_name,
            "rt/parameter_events"
        );
        assert_eq!(
            ROS2_BUILTIN_PARAMETER_EVENTS.type_name,
            "rcl_interfaces::msg::dds_::ParameterEvent_"
        );
    }

    #[test]
    fn default_endpoint_set_has_six_bits() {
        // Verify 6 bits are set
        assert_eq!(ROS2_DEFAULT_ENDPOINT_SET.count_ones(), 6);
    }
}
