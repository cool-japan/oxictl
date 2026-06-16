use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;

pub const PROTOCOL_VERSION_2_3: ProtocolVersion = ProtocolVersion { major: 2, minor: 3 };

/// OXICTL development vendor ID. Not registered with OMG; for internal use only.
pub const VENDOR_ID_OXICTL: VendorId = VendorId([0x01, 0x10]);

pub const GUID_UNKNOWN: Guid = Guid {
    prefix: GuidPrefix([0u8; 12]),
    entity_id: ENTITYID_UNKNOWN,
};
pub const ENTITYID_UNKNOWN: EntityId = EntityId {
    entity_key: [0, 0, 0],
    entity_kind: 0x00,
};
pub const ENTITYID_PARTICIPANT: EntityId = EntityId {
    entity_key: [0, 0, 0x01],
    entity_kind: 0xC1,
};

// SPDP builtin participant writer/reader
pub const ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER: EntityId = EntityId {
    entity_key: [0, 0x01, 0x00],
    entity_kind: 0xC2,
};
pub const ENTITYID_SPDP_BUILTIN_PARTICIPANT_READER: EntityId = EntityId {
    entity_key: [0, 0x01, 0x00],
    entity_kind: 0xC7,
};

// SEDP builtin endpoints writer/reader (topics, publications, subscriptions)
pub const ENTITYID_SEDP_BUILTIN_TOPICS_WRITER: EntityId = EntityId {
    entity_key: [0, 0, 0x02],
    entity_kind: 0xC2,
};
pub const ENTITYID_SEDP_BUILTIN_TOPICS_READER: EntityId = EntityId {
    entity_key: [0, 0, 0x02],
    entity_kind: 0xC7,
};
pub const ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER: EntityId = EntityId {
    entity_key: [0, 0, 0x03],
    entity_kind: 0xC2,
};
pub const ENTITYID_SEDP_BUILTIN_PUBLICATIONS_READER: EntityId = EntityId {
    entity_key: [0, 0, 0x03],
    entity_kind: 0xC7,
};
pub const ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER: EntityId = EntityId {
    entity_key: [0, 0, 0x04],
    entity_kind: 0xC2,
};
pub const ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_READER: EntityId = EntityId {
    entity_key: [0, 0, 0x04],
    entity_kind: 0xC7,
};

/// 12-byte GUID prefix (identifies a participant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidPrefix(pub [u8; 12]);

impl GuidPrefix {
    pub fn is_unknown(&self) -> bool {
        self.0 == [0u8; 12]
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let bytes: [u8; 12] = cur
            .read_bytes(12)?
            .try_into()
            .map_err(|_| RtpsError::TruncatedHeader)?;
        Ok(Self(bytes))
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_bytes(&self.0)
    }
}

/// 4-byte entity identifier (3-byte key + 1-byte kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityId {
    pub entity_key: [u8; 3],
    pub entity_kind: u8,
}

impl EntityId {
    pub fn is_unknown(&self) -> bool {
        self == &ENTITYID_UNKNOWN
    }

    /// True if this entity is a writer per RTPS Table 9.1.
    pub fn is_writer(&self) -> bool {
        matches!(self.entity_kind & 0x0F, 0x02 | 0x03 | 0x04 | 0x07)
            && (self.entity_kind & 0xC0 == 0 || self.entity_kind == 0xC2)
    }

    /// True if this entity is a reader per RTPS Table 9.1.
    pub fn is_reader(&self) -> bool {
        matches!(self.entity_kind & 0x0F, 0x04 | 0x07) || self.entity_kind == 0xC7
    }

    /// True if this entity is a built-in RTPS entity.
    pub fn is_builtin(&self) -> bool {
        self.entity_kind & 0xC0 == 0xC0
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let key: [u8; 3] = cur
            .read_bytes(3)?
            .try_into()
            .map_err(|_| RtpsError::TruncatedHeader)?;
        let kind = cur.read_u8()?;
        Ok(Self {
            entity_key: key,
            entity_kind: kind,
        })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_bytes(&self.entity_key)?;
        w.write_u8(self.entity_kind)
    }
}

/// Global unique identifier for a participant or endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub prefix: GuidPrefix,
    pub entity_id: EntityId,
}

impl Guid {
    pub fn new(prefix: GuidPrefix, entity_id: EntityId) -> Self {
        Self { prefix, entity_id }
    }

    pub fn is_unknown(&self) -> bool {
        self == &GUID_UNKNOWN
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let prefix = GuidPrefix::parse(cur)?;
        let entity_id = EntityId::parse(cur)?;
        Ok(Self { prefix, entity_id })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.prefix.serialize(w)?;
        self.entity_id.serialize(w)
    }
}

/// 2-byte vendor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VendorId(pub [u8; 2]);

impl VendorId {
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let bytes: [u8; 2] = cur
            .read_bytes(2)?
            .try_into()
            .map_err(|_| RtpsError::TruncatedHeader)?;
        Ok(Self(bytes))
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_bytes(&self.0)
    }
}

/// RTPS protocol version (major.minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolVersion {
    pub major: u8,
    pub minor: u8,
}

impl ProtocolVersion {
    pub fn is_compatible_2x(&self) -> bool {
        self.major == 2
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let major = cur.read_u8()?;
        let minor = cur.read_u8()?;
        Ok(Self { major, minor })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u8(self.major)?;
        w.write_u8(self.minor)
    }
}
