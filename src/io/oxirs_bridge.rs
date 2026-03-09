//! oxirs knowledge graph bridge (stub).
//!
//! Integration point for connecting oxictl to the oxirs knowledge graph
//! and symbolic reasoning ecosystem.
//!
//! Exposes a trait for querying and updating symbolic state (e.g. equipment
//! status, process tags, alarm states) from within control loops.

/// A symbolic tag value — scalar, boolean, or enumerated.
#[derive(Debug, Clone)]
pub enum TagValue {
    /// Floating-point measurement.
    Real(f64),
    /// Boolean flag.
    Bool(bool),
    /// Integer enum (maps to named state in the knowledge graph).
    Enum(u32),
    /// Unavailable / unknown.
    Unavailable,
}

/// A knowledge graph tag descriptor.
#[derive(Debug, Clone)]
pub struct Tag {
    pub id: u32,
    pub name: &'static str,
    pub value: TagValue,
    pub timestamp_us: u64,
}

/// Trait for oxirs-compatible knowledge graph interfaces.
pub trait OxirsInterface {
    /// Query a tag by ID. Returns None if not found.
    fn query_tag(&self, tag_id: u32) -> Option<&Tag>;

    /// Update a tag value.
    fn update_tag(&mut self, tag_id: u32, value: TagValue, timestamp_us: u64) -> bool;

    /// Check if the knowledge graph is available.
    fn is_available(&self) -> bool;
}

/// Stub oxirs interface backed by a fixed-size tag array.
pub struct NullOxirsInterface<const N: usize> {
    tags: [Option<Tag>; N],
}

impl<const N: usize> NullOxirsInterface<N> {
    pub fn new() -> Self {
        Self {
            tags: core::array::from_fn(|_| None),
        }
    }

    /// Pre-populate a tag slot.
    pub fn define_tag(&mut self, slot: usize, id: u32, name: &'static str) {
        if slot < N {
            self.tags[slot] = Some(Tag {
                id,
                name,
                value: TagValue::Unavailable,
                timestamp_us: 0,
            });
        }
    }
}

impl<const N: usize> Default for NullOxirsInterface<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> OxirsInterface for NullOxirsInterface<N> {
    fn query_tag(&self, tag_id: u32) -> Option<&Tag> {
        self.tags
            .iter()
            .filter_map(|t| t.as_ref())
            .find(|t| t.id == tag_id)
    }

    fn update_tag(&mut self, tag_id: u32, value: TagValue, timestamp_us: u64) -> bool {
        for tag in self.tags.iter_mut().flatten() {
            if tag.id == tag_id {
                tag.value = value;
                tag.timestamp_us = timestamp_us;
                return true;
            }
        }
        false
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn define_and_query_tag() {
        let mut iface = NullOxirsInterface::<8>::new();
        iface.define_tag(0, 42, "temperature");
        let tag = iface.query_tag(42).unwrap();
        assert_eq!(tag.name, "temperature");
        assert!(matches!(tag.value, TagValue::Unavailable));
    }

    #[test]
    fn update_tag_value() {
        let mut iface = NullOxirsInterface::<4>::new();
        iface.define_tag(0, 1, "pressure");
        assert!(iface.update_tag(1, TagValue::Real(101.325), 1000));
        let tag = iface.query_tag(1).unwrap();
        if let TagValue::Real(v) = tag.value {
            assert!((v - 101.325).abs() < 1e-10);
        } else {
            panic!("Expected Real");
        }
    }

    #[test]
    fn query_missing_tag_returns_none() {
        let iface = NullOxirsInterface::<4>::new();
        assert!(iface.query_tag(999).is_none());
    }

    #[test]
    fn update_missing_tag_returns_false() {
        let mut iface = NullOxirsInterface::<4>::new();
        assert!(!iface.update_tag(999, TagValue::Bool(true), 0));
    }
}
