use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::fragment::FragmentNumber;
use crate::protocol::dds::types::guid::EntityId;
use crate::protocol::dds::types::parameter::ParameterList;
use crate::protocol::dds::types::sequence::SequenceNumber;

/// DATA submessage body (RTPS 2.3 Section 8.3.7.2).
///
/// Flags: E(0) Q(1) D(2) K(3) N(4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Data<'a> {
    pub endianness: Endianness,
    pub inline_qos_flag: bool,
    pub data_flag: bool,
    pub key_flag: bool,
    pub non_standard_payload_flag: bool,
    pub extra_flags: u16,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub writer_sn: SequenceNumber,
    pub inline_qos: Option<ParameterList<'a>>,
    pub serialized_payload: &'a [u8],
}

impl<'a> Data<'a> {
    pub fn parse(flags: u8, body: &'a [u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let inline_qos_flag = flags & 0x02 != 0;
        let data_flag = flags & 0x04 != 0;
        let key_flag = flags & 0x08 != 0;
        let non_standard_payload_flag = flags & 0x10 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let extra_flags = cur.read_u16()?;
        let octets_to_inline_qos = cur.read_u16()? as usize;

        // pos_after_onh is the position right after the two header u16 fields (= 4)
        let pos_after_onh = cur.position();

        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let writer_sn = SequenceNumber::parse(&mut cur)?;

        // Forward-compat skip: if octets_to_inline_qos > 16, skip the extra bytes
        let bytes_consumed = cur.position() - pos_after_onh;
        if bytes_consumed < octets_to_inline_qos {
            cur.skip(octets_to_inline_qos - bytes_consumed)?;
        }

        let inline_qos = if inline_qos_flag {
            Some(ParameterList::parse(&mut cur)?)
        } else {
            None
        };

        let serialized_payload = if data_flag || key_flag {
            cur.peek_remaining()
        } else {
            &[]
        };

        Ok(Self {
            endianness,
            inline_qos_flag,
            data_flag,
            key_flag,
            non_standard_payload_flag,
            extra_flags,
            reader_id,
            writer_id,
            writer_sn,
            inline_qos,
            serialized_payload,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u16(self.extra_flags)?;
        w.write_u16(16)?; // octets_to_inline_qos for standard DATA
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.writer_sn.serialize(w)?;
        if let Some(ref iqos) = self.inline_qos {
            iqos.serialize(w)?;
        }
        if self.data_flag || self.key_flag {
            w.write_bytes(self.serialized_payload)?;
        }
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.inline_qos_flag {
            f |= 0x02;
        }
        if self.data_flag {
            f |= 0x04;
        }
        if self.key_flag {
            f |= 0x08;
        }
        if self.non_standard_payload_flag {
            f |= 0x10;
        }
        f
    }
}

/// DATA_FRAG submessage body (RTPS 2.3 Section 8.3.7.3).
///
/// Flags: E(0) Q(1) K(3) N(4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrag<'a> {
    pub endianness: Endianness,
    pub inline_qos_flag: bool,
    pub key_flag: bool,
    pub non_standard_payload_flag: bool,
    pub extra_flags: u16,
    pub reader_id: EntityId,
    pub writer_id: EntityId,
    pub writer_sn: SequenceNumber,
    pub fragment_starting_num: FragmentNumber,
    pub fragments_in_submessage: u16,
    pub fragment_size: u16,
    pub sample_size: u32,
    pub inline_qos: Option<ParameterList<'a>>,
    pub serialized_payload: &'a [u8],
}

impl<'a> DataFrag<'a> {
    pub fn parse(flags: u8, body: &'a [u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let inline_qos_flag = flags & 0x02 != 0;
        let key_flag = flags & 0x08 != 0;
        let non_standard_payload_flag = flags & 0x10 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let extra_flags = cur.read_u16()?;
        let octets_to_inline_qos = cur.read_u16()? as usize;

        let pos_after_onh = cur.position();

        let reader_id = EntityId::parse(&mut cur)?;
        let writer_id = EntityId::parse(&mut cur)?;
        let writer_sn = SequenceNumber::parse(&mut cur)?;
        let fragment_starting_num = FragmentNumber::parse(&mut cur)?;
        let fragments_in_submessage = cur.read_u16()?;
        let fragment_size = cur.read_u16()?;
        let sample_size = cur.read_u32()?;

        // Forward-compat skip for octets_to_inline_qos = 28 standard
        let bytes_consumed = cur.position() - pos_after_onh;
        if bytes_consumed < octets_to_inline_qos {
            cur.skip(octets_to_inline_qos - bytes_consumed)?;
        }

        let inline_qos = if inline_qos_flag {
            Some(ParameterList::parse(&mut cur)?)
        } else {
            None
        };

        let serialized_payload = cur.peek_remaining();

        Ok(Self {
            endianness,
            inline_qos_flag,
            key_flag,
            non_standard_payload_flag,
            extra_flags,
            reader_id,
            writer_id,
            writer_sn,
            fragment_starting_num,
            fragments_in_submessage,
            fragment_size,
            sample_size,
            inline_qos,
            serialized_payload,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u16(self.extra_flags)?;
        w.write_u16(28)?; // octets_to_inline_qos for DATA_FRAG = 28
        self.reader_id.serialize(w)?;
        self.writer_id.serialize(w)?;
        self.writer_sn.serialize(w)?;
        self.fragment_starting_num.serialize(w)?;
        w.write_u16(self.fragments_in_submessage)?;
        w.write_u16(self.fragment_size)?;
        w.write_u32(self.sample_size)?;
        if let Some(ref iqos) = self.inline_qos {
            iqos.serialize(w)?;
        }
        w.write_bytes(self.serialized_payload)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.inline_qos_flag {
            f |= 0x02;
        }
        if self.key_flag {
            f |= 0x08;
        }
        if self.non_standard_payload_flag {
            f |= 0x10;
        }
        f
    }
}
