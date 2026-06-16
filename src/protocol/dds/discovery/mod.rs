//! SPDP and SEDP discovery for RTPS 2.3.
//!
//! - SPDP (Simple Participant Discovery Protocol): participant announcement and discovery.
//! - SEDP (Simple Endpoint Discovery Protocol): endpoint (publication/subscription) discovery.
//!
//! Requires `dds-discovery` feature (pulls in `dds-transport`).

pub mod endpoint_data;
pub mod error;
pub mod participant_data;
pub mod qos;
pub mod qos_match;
pub mod qos_profile;
pub mod sedp;
pub mod spdp;

pub use endpoint_data::{PublicationBuiltinTopicData, SubscriptionBuiltinTopicData};
pub use error::DiscoveryError;
pub use participant_data::{
    ParticipantBuiltinTopicData, BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER,
    BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR, BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER,
    BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR, BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER,
    BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR,
};
pub use qos::{
    DeadlineQosPolicy, DestinationOrderKind, DestinationOrderQosPolicy, DurabilityKind,
    DurabilityQosPolicy, HistoryKind, HistoryQosPolicy, LifespanQosPolicy, LivelinessKind,
    LivelinessQosPolicy, OwnershipKind, OwnershipQosPolicy, OwnershipStrengthQosPolicy,
    ReliabilityKind, ReliabilityQosPolicy, ResourceLimitsQosPolicy,
};
pub use qos_match::{match_endpoint_qos, match_endpoint_qos_extended, IncompatibleQos};
pub use qos_profile::QosProfile;
pub use sedp::{IncomingResult, SedpParticipant};
pub use spdp::{DiscoveredParticipant, SpdpParticipant};
