//! `ServiceServer<S>` — request/reply server for a ROS2 service.

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::api::publisher::Publisher;
use crate::protocol::dds::api::subscription::Subscription;

use super::wrappers::{ReplyWrapper, RequestWrapper, Service};

/// Type-safe DDS service server.
///
/// Created by [`create_server`](super::create_server).  Call `process` once
/// per spin cycle to handle pending requests: it drains the request queue,
/// applies `handler` to each request, and publishes the correlated reply.
pub struct ServiceServer<S: Service> {
    request_sub: Subscription<RequestWrapper<S>>,
    reply_pub: Publisher<ReplyWrapper<S>>,
}

impl<S: Service> ServiceServer<S> {
    pub(super) fn new(
        request_sub: Subscription<RequestWrapper<S>>,
        reply_pub: Publisher<ReplyWrapper<S>>,
    ) -> Self {
        Self {
            request_sub,
            reply_pub,
        }
    }

    /// Handle all pending requests with `handler`.
    ///
    /// For each request: deserializes the body, calls `handler(&request)`,
    /// echoes the `SampleIdentity` header into the reply, and publishes it.
    /// Returns the number of requests processed.
    pub fn process<F>(
        &mut self,
        participant: &mut Participant,
        mut handler: F,
    ) -> Result<usize, DdsApiError>
    where
        F: FnMut(&S::Request) -> S::Response,
    {
        let samples = participant.take(&self.request_sub);
        let count = samples.len();
        for sample in samples {
            let RequestWrapper { header, body } = sample.data;
            let response = handler(&body);
            // Echo the request SampleIdentity verbatim — this IS the correlation.
            let reply = ReplyWrapper::<S> {
                header,
                body: response,
            };
            participant.publish(&self.reply_pub, &reply)?;
        }
        Ok(count)
    }
}
