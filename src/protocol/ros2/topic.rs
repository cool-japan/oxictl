//! ROS2 topic pub/sub bridge stub.
//!
//! Provides a no_std-compatible message buffer that bridges between
//! oxictl control data and the ROS2 DDS middleware.
//!
//! In production, this would use the `rclrs` or `ros2_rust` crate.
//! Here we implement a ring-buffer message queue for integration testing.

use heapless::spsc::{Consumer, Producer, Queue};

/// A ROS2-compatible message: timestamped generic payload.
#[derive(Debug, Clone, Copy)]
pub struct RosMessage<T: Copy> {
    /// Monotonic timestamp (ns since boot).
    pub stamp_ns: u64,
    /// Message payload.
    pub payload: T,
}

/// ROS2 topic publisher (producer side of a SPSC queue).
pub struct Publisher<'q, T: Copy, const CAP: usize> {
    pub topic: &'static str,
    producer: Producer<'q, RosMessage<T>>,
    pub publish_count: u64,
}

impl<'q, T: Copy, const CAP: usize> Publisher<'q, T, CAP> {
    /// Publish a message. Returns false if queue is full.
    pub fn publish(&mut self, stamp_ns: u64, payload: T) -> bool {
        let msg = RosMessage { stamp_ns, payload };
        if self.producer.enqueue(msg).is_ok() {
            self.publish_count += 1;
            true
        } else {
            false
        }
    }
}

/// ROS2 topic subscriber (consumer side of a SPSC queue).
pub struct Subscriber<'q, T: Copy, const CAP: usize> {
    pub topic: &'static str,
    consumer: Consumer<'q, RosMessage<T>>,
    pub receive_count: u64,
}

impl<'q, T: Copy, const CAP: usize> Subscriber<'q, T, CAP> {
    /// Try to receive a message. Returns None if queue is empty.
    pub fn try_recv(&mut self) -> Option<RosMessage<T>> {
        self.consumer.dequeue().map(|msg| {
            self.receive_count += 1;
            msg
        })
    }
}

/// Create a matched publisher/subscriber pair sharing a queue.
pub fn create_topic<'a, T: Copy, const CAP: usize>(
    topic: &'static str,
    queue: &'a mut Queue<RosMessage<T>, CAP>,
) -> (Publisher<'a, T, CAP>, Subscriber<'a, T, CAP>) {
    let (producer, consumer) = queue.split();
    (
        Publisher {
            topic,
            producer,
            publish_count: 0,
        },
        Subscriber {
            topic,
            consumer,
            receive_count: 0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_pub_sub() {
        let mut queue: Queue<RosMessage<f64>, 8> = Queue::new();
        let (mut pub_, mut sub) = create_topic::<f64, 8>("sensor/temperature", &mut queue);

        pub_.publish(1000, 25.3);
        pub_.publish(2000, 25.5);

        let msg = sub.try_recv().unwrap();
        assert_eq!(msg.stamp_ns, 1000);
        assert!((msg.payload - 25.3).abs() < 1e-10);
        assert_eq!(pub_.publish_count, 2);
        assert_eq!(sub.receive_count, 1);
    }

    #[test]
    fn topic_full_queue() {
        // heapless SPSC Queue<T, N> holds N-1 elements; use N=3 to hold 2
        let mut queue: Queue<RosMessage<u32>, 3> = Queue::new();
        let (mut pub_, _sub) = create_topic::<u32, 3>("cmd", &mut queue);
        assert!(pub_.publish(0, 1));
        assert!(pub_.publish(1, 2));
        assert!(!pub_.publish(2, 3)); // queue full
    }

    #[test]
    fn topic_empty_subscriber() {
        let mut queue: Queue<RosMessage<i32>, 4> = Queue::new();
        let (_pub_, mut sub) = create_topic::<i32, 4>("ctrl", &mut queue);
        assert!(sub.try_recv().is_none());
    }
}
