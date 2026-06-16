use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::guid::EntityId;
use crate::protocol::dds::types::sequence::{SequenceNumber, SequenceNumberSet};

/// GAP submessage body (RTPS 2.3 Section 8.3.7.4).
///
/// Flags: E(0) G(3:group_info)
/// Body: reader_id(4) + writer_id(4) + gap_start(8) + gap_list(variable)
/// G flag extension is not parsed in Phase 22.1; group_info_flag is stored only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub endianness: Endianness,
    pub group_info_flag: bool,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub gap_start: SequenceNumber,
    pub gap_list: SequenceNumberSet,
}

impl Gap {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let group_info_flag = flags & 0x08 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let gap_start = SequenceNumber::parse(&mut cur)?;
        let gap_list = SequenceNumberSet::parse(&mut cur)?;

        Ok(Self {
            endianness,
            group_info_flag,
            reader_id,
            writer_id,
            gap_start,
            gap_list,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.gap_start.serialize(w)?;
        self.gap_list.serialize(w)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.group_info_flag {
            f |= 0x08;
        }
        f
    }
}
