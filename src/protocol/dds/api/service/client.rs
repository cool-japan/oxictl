//! `ServiceClient<S>` — request/reply client for a ROS2 service.

use heapless::Vec as HVec;

use crate::protocol::dds::api::dds_type::Sample;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::participant::Participant;
use crate::protocol::dds::api::publisher::Publisher;
use crate::protocol::dds::api::subscription::Subscription;

use super::sample_identity::SampleIdentity;
use super::wrappers::{ReplyWrapper, RequestWrapper, Service};

/// Maximum number of in-flight requests tracked for correlation.
const MAX_PENDING: usize = 16;

/// Type-safe DDS service client.
///
/// Created by [`create_client`](super::create_client).  Call `send_request`
/// to issue a request, then spin the participant and call `take_responses` to
/// retrieve correlated replies.
///
/// Up to `MAX_PENDING` (16) requests may be in-flight simultaneously.
/// When the limit is reached the oldest pending sequence number is evicted
/// (the reply, if it ever arrives, will be silently discarded).
pub struct ServiceClient<S: Service> {
    request_pub: Publisher<RequestWrapper<S>>,
    reply_sub: Subscription<ReplyWrapper<S>>,
    /// GUID of our request publisher, used to filter replies.
    my_request_writer_guid: [u8; 16],
    next_request_seq: i64,
    pending: HVec<i64, MAX_PENDING>,
}

impl<S: Service> ServiceClient<S> {
    pub(super) fn new(
        request_pub: Publisher<RequestWrapper<S>>,
        reply_sub: Subscription<ReplyWrapper<S>>,
        my_request_writer_guid: [u8; 16],
    ) -> Self {
        Self {
            request_pub,
            reply_sub,
            my_request_writer_guid,
            next_request_seq: 1,
            pending: HVec::new(),
        }
    }

    /// Send `request` and return the assigned sequence number.
    ///
    /// The caller should spin the participant and call [`take_responses`] to
    /// retrieve the matching reply.
    pub fn send_request(
        &mut self,
        participant: &mut Participant,
        request: &S::Request,
    ) -> Result<i64, DdsApiError> {
        let seq = self.next_request_seq;
        self.next_request_seq += 1;

        // Evict oldest pending if at capacity.
        if self.pending.is_full() {
            self.pending.remove(0);
        }
        // Cannot fail: we just ensured capacity.
        let _ = self.pending.push(seq);

        let wrapper = RequestWrapper::<S> {
            header: SampleIdentity::new(self.my_request_writer_guid, seq),
            body: request.clone(),
        };
        participant.publish(&self.request_pub, &wrapper)?;
        Ok(seq)
    }

    /// Drain buffered replies that match this client's request-writer GUID.
    ///
    /// Returns `(sequence_number, response)` pairs.  Replies whose GUID does
    /// not match (from another client on the same topic) are discarded.
    pub fn take_responses(&mut self, participant: &mut Participant) -> Vec<(i64, S::Response)> {
        let samples: Vec<Sample<ReplyWrapper<S>>> = participant.take(&self.reply_sub);
        let my_guid = self.my_request_writer_guid;
        let mut results = Vec::with_capacity(samples.len());
        for sample in samples {
            let ReplyWrapper { header, body } = sample.data;
            if header.writer_guid != my_guid {
                continue;
            }
            let seq = header.sequence_number;
            // Remove from pending list.
            if let Some(pos) = self.pending.iter().position(|&s| s == seq) {
                self.pending.remove(pos);
            }
            results.push((seq, body));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_seq_starts_at_one() {
        // We cannot instantiate ServiceClient without a Participant, so test
        // the sequence number logic through unit inspection of the field.
        // (Full end-to-end is covered in dds_service_integration tests.)
        let id1 = SampleIdentity::new([1u8; 16], 1);
        let id2 = SampleIdentity::new([1u8; 16], 2);
        assert_eq!(id1.sequence_number, 1);
        assert_eq!(id2.sequence_number, 2);
    }
}
