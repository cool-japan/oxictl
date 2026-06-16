//! ROS2 service layer — type-safe request/reply over DDS.
//!
//! Provides [`ServiceClient<S>`] and [`ServiceServer<S>`] for request/reply RPC
//! on top of the reliable DDS endpoint layer.
//!
//! # Quickstart
//! ```rust,ignore
//! use oxictl::protocol::dds::api::service::{create_client, create_server};
//! use oxictl::protocol::dds::api::{AddTwoInts, ...};
//! let mut server = create_server::<AddTwoInts>(&mut participant, "add_two_ints", &qos)?;
//! let mut client = create_client::<AddTwoInts>(&mut participant, "add_two_ints", &qos)?;
//! // ... participant.spin_once() ...
//! let seq = client.send_request(&mut participant, &request)?;
//! server.process(&mut participant, |req| response_for(req))?;
//! let replies = client.take_responses(&mut participant);
//! ```

pub mod client;
pub mod sample_identity;
pub mod server;
pub mod wrappers;

pub use client::ServiceClient;
pub use sample_identity::SampleIdentity;
pub use server::ServiceServer;
pub use wrappers::{ReplyWrapper, RequestWrapper, Service, ServiceField};

use heapless::String as HString;

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::ros2::topic_naming::{encode_topic_name, Ros2TopicKind};

/// Create a [`ServiceClient<S>`] for the named service.
///
/// Registers a request publisher and a reply subscription with the participant.
/// The client captures its request-publisher GUID for reply correlation.
pub fn create_client<S: Service>(
    participant: &mut Participant,
    service_name: &str,
    qos: &QosProfile,
) -> Result<ServiceClient<S>, DdsApiError> {
    let mut req_topic = HString::<256>::new();
    encode_topic_name(&mut req_topic, service_name, Ros2TopicKind::ServiceRequest)
        .map_err(|_| DdsApiError::Serialization("service name too long or invalid"))?;

    let mut rep_topic = HString::<256>::new();
    encode_topic_name(&mut rep_topic, service_name, Ros2TopicKind::ServiceReply)
        .map_err(|_| DdsApiError::Serialization("service name too long or invalid"))?;

    let req_pub = participant.create_publisher::<RequestWrapper<S>>(req_topic.as_str(), qos)?;
    let req_guid = participant
        .publisher_guid(&req_pub)
        .ok_or(DdsApiError::Serialization("publisher guid unavailable"))?;
    let rep_sub = participant.create_subscription::<ReplyWrapper<S>>(rep_topic.as_str(), qos)?;

    Ok(ServiceClient::new(req_pub, rep_sub, req_guid))
}

/// Create a [`ServiceServer<S>`] for the named service.
///
/// Registers a request subscription and a reply publisher with the participant.
/// Call [`ServiceServer::process`] once per spin cycle to handle requests.
pub fn create_server<S: Service>(
    participant: &mut Participant,
    service_name: &str,
    qos: &QosProfile,
) -> Result<ServiceServer<S>, DdsApiError> {
    let mut req_topic = HString::<256>::new();
    encode_topic_name(&mut req_topic, service_name, Ros2TopicKind::ServiceRequest)
        .map_err(|_| DdsApiError::Serialization("service name too long or invalid"))?;

    let mut rep_topic = HString::<256>::new();
    encode_topic_name(&mut rep_topic, service_name, Ros2TopicKind::ServiceReply)
        .map_err(|_| DdsApiError::Serialization("service name too long or invalid"))?;

    let req_sub = participant.create_subscription::<RequestWrapper<S>>(req_topic.as_str(), qos)?;
    let rep_pub = participant.create_publisher::<ReplyWrapper<S>>(rep_topic.as_str(), qos)?;

    Ok(ServiceServer::new(req_sub, rep_pub))
}
