//! Atomic entity-ID allocator for unique user endpoints within one process.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::protocol::dds::types::guid::EntityId;

/// Process-wide counter ensuring uniqueness across all `Participant` instances.
///
/// Using `Ordering::Relaxed` is safe here: we only need each fetch-and-increment
/// to be atomic (no ordering guarantees with respect to other memory accesses are
/// needed — only uniqueness of the counter value matters).
static ENTITY_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Allocates unique `EntityId` values within one process.
///
/// Uses a process-global atomic counter to ensure each allocated ID is unique
/// regardless of which `Participant` instance calls `next_writer` / `next_reader`.
/// The first 3 bytes of the entity key encode the counter value in little-endian;
/// the kind byte distinguishes writers from readers.
pub struct EntityIdAllocator;

impl EntityIdAllocator {
    /// Allocate a fresh writer-with-key `EntityId` (kind = `0x02`).
    pub fn next_writer() -> EntityId {
        let n = ENTITY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let bytes = n.to_le_bytes(); // [b0, b1, b2, b3]
        EntityId {
            entity_key: [bytes[0], bytes[1], bytes[2]],
            entity_kind: 0x02, // user-defined writer with key
        }
    }

    /// Allocate a fresh reader-with-key `EntityId` (kind = `0x07`).
    pub fn next_reader() -> EntityId {
        let n = ENTITY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let bytes = n.to_le_bytes();
        EntityId {
            entity_key: [bytes[0], bytes[1], bytes[2]],
            entity_kind: 0x07, // user-defined reader with key
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_ids_are_unique() {
        let a = EntityIdAllocator::next_writer();
        let b = EntityIdAllocator::next_writer();
        assert_ne!(a.entity_key, b.entity_key);
        assert_eq!(a.entity_kind, 0x02);
        assert_eq!(b.entity_kind, 0x02);
    }

    #[test]
    fn reader_ids_are_unique() {
        let a = EntityIdAllocator::next_reader();
        let b = EntityIdAllocator::next_reader();
        assert_ne!(a.entity_key, b.entity_key);
        assert_eq!(a.entity_kind, 0x07);
    }

    #[test]
    fn writer_reader_kind_differs() {
        let w = EntityIdAllocator::next_writer();
        let r = EntityIdAllocator::next_reader();
        assert_eq!(w.entity_kind, 0x02);
        assert_eq!(r.entity_kind, 0x07);
    }
}
