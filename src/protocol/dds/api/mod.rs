//! High-level type-safe DDS API (`dds-api` feature).
//!
//! Provides `Publisher<T>`, `Subscription<T>`, and `Participant` built on top of
//! the Phase 22 RTPS/DDS stack.  Requires the `dds-api` feature which pulls in
//! `dds-ros2` → `dds-stateful` → `dds-discovery` → `dds-transport` → `dds`.
//!
//! # Quickstart
//! ```rust,ignore
//! use oxictl::protocol::dds::api::{DdsType, Participant, QosProfile};
//! // (implement DdsType for your message type, then:)
//! let mut p = Participant::new(guid_prefix, QosProfile::ros2_default())?;
//! let pub1 = p.create_publisher::<MyMsg>("my_topic", &QosProfile::ros2_default())?;
//! p.publish(&pub1, &MyMsg::default())?;
//! loop { p.spin_once()?; }
//! ```

pub mod action;
pub mod builtin_impls;
pub mod dds_type;
pub mod entity_id;
pub mod error;
pub mod participant;
pub mod publisher;
pub mod service;
pub mod subscription;

pub use action::{
    create_action_client, create_action_server, Action, ActionClient, ActionHandler, ActionServer,
    ExecuteResult, GetResultRequest, GetResultResponse, GetResultService, GoalOutcome,
    SendGoalRequest, SendGoalResponse, SendGoalService,
};
pub use builtin_impls::{LogOwned, ParameterEventOwned};
pub use dds_type::{DdsType, Sample};
pub use entity_id::EntityIdAllocator;
pub use error::DdsApiError;
pub use participant::Participant;
pub use publisher::Publisher;
pub use service::{
    create_client, create_server, ReplyWrapper, RequestWrapper, SampleIdentity, Service,
    ServiceClient, ServiceField, ServiceServer,
};
pub use subscription::Subscription;
