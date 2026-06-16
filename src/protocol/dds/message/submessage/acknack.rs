use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::fragment::FragmentNumberSet;
use crate::protocol::dds::types::guid::EntityId;
use crate::protocol::dds::types::sequence::{SequenceNumber, SequenceNumberSet};

/// ACKNACK submessage body (RTPS 2.3 Section 8.3.7.1).
///
/// Flags: E(0) F(1:final)
/// Body: reader_id(4) + writer_id(4) + reader_sn_state(variable) + count(4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckNack {
    pub endianness: Endianness,
    pub final_flag: bool,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub reader_sn_state: SequenceNumberSet,
    pub count: i32,
}

impl AckNack {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let final_flag = flags & 0x02 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let reader_sn_state = SequenceNumberSet::parse(&mut cur)?;
        let count = cur.read_i32()?;

        Ok(Self {
            endianness,
            final_flag,
            reader_id,
            writer_id,
            reader_sn_state,
            count,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.reader_sn_state.serialize(w)?;
        w.write_i32(self.count)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.final_flag {
            f |= 0x02;
        }
        f
    }
}

/// NACK_FRAG submessage body (RTPS 2.3 Section 8.3.7.11).
///
/// Body: reader_id(4) + writer_id(4) + writer_sn(8) + fragment_number_state(variable) + count(4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NackFrag {
    pub endianness: Endianness,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub writer_sn: SequenceNumber,
    pub fragment_number_state: FragmentNumberSet,
    pub count: i32,
}

impl NackFrag {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let mut cur = ByteCursor::new(body, endianness);
        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let writer_sn = SequenceNumber::parse(&mut cur)?;
        let fragment_number_state = FragmentNumberSet::parse(&mut cur)?;
        let count = cur.read_i32()?;

        Ok(Self {
            endianness,
            reader_id,
            writer_id,
            writer_sn,
            fragment_number_state,
            count,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.writer_sn.serialize(w)?;
        self.fragment_number_state.serialize(w)?;
        w.write_i32(self.count)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        self.endianness.into_flags(0u8)
    }
}
