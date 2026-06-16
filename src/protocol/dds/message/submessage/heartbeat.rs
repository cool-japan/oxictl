use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::fragment::FragmentNumber;
use crate::protocol::dds::types::guid::EntityId;
use crate::protocol::dds::types::sequence::SequenceNumber;

/// HEARTBEAT submessage body (RTPS 2.3 Section 8.3.7.5).
///
/// Flags: E(0) F(1:final) L(2:liveliness) G(3:group_info)
/// Body: reader_id(4) + writer_id(4) + first_sn(8) + last_sn(8) + count(4) = 28 bytes.
/// G flag extension is not parsed in Phase 22.1; group_info_flag is stored only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heartbeat {
    pub endianness: Endianness,
    pub final_flag: bool,
    pub liveliness_flag: bool,
    pub group_info_flag: bool,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub first_sn: SequenceNumber,
    pub last_sn: SequenceNumber,
    pub count: i32,
}

impl Heartbeat {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let final_flag = flags & 0x02 != 0;
        let liveliness_flag = flags & 0x04 != 0;
        let group_info_flag = flags & 0x08 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let first_sn = SequenceNumber::parse(&mut cur)?;
        let last_sn = SequenceNumber::parse(&mut cur)?;
        let count = cur.read_i32()?;

        Ok(Self {
            endianness,
            final_flag,
            liveliness_flag,
            group_info_flag,
            reader_id,
            writer_id,
            first_sn,
            last_sn,
            count,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.first_sn.serialize(w)?;
        self.last_sn.serialize(w)?;
        w.write_i32(self.count)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.final_flag {
            f |= 0x02;
        }
        if self.liveliness_flag {
            f |= 0x04;
        }
        if self.group_info_flag {
            f |= 0x08;
        }
        f
    }
}

/// HEARTBEAT_FRAG submessage body (RTPS 2.3 Section 8.3.7.6).
///
/// Body: reader_id(4) + writer_id(4) + writer_sn(8) + last_fragment_num(4) + count(4) = 24 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeartbeatFrag {
    pub endianness: Endianness,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub writer_sn: SequenceNumber,
    pub last_fragment_num: FragmentNumber,
    pub count: i32,
}

impl HeartbeatFrag {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let mut cur = ByteCursor::new(body, endianness);
        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let writer_sn = SequenceNumber::parse(&mut cur)?;
        let last_fragment_num = FragmentNumber::parse(&mut cur)?;
        let count = cur.read_i32()?;

        Ok(Self {
            endianness,
            reader_id,
            writer_id,
            writer_sn,
            last_fragment_num,
            count,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.writer_sn.serialize(w)?;
        self.last_fragment_num.serialize(w)?;
        w.write_i32(self.count)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        self.endianness.into_flags(0u8)
    }
}
