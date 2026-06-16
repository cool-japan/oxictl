//! Integration tests for the RTPS 2.3 wire-protocol parser and serializer.

use crate::protocol::dds::{
    byte_cursor::Endianness,
    error::RtpsError,
    message::{
        submessage::{
            AckNack, Data, DataFrag, Gap, Heartbeat, HeartbeatFrag, InfoDestination, InfoReply,
            InfoReplyIp4, InfoSource, InfoTimestamp, NackFrag,
        },
        Message, MessageHeader, Submessage,
    },
    parse_message, serialize_message,
    types::{
        fragment::{FragmentNumber, FragmentNumberSet},
        guid::{
            EntityId, GuidPrefix, ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER, ENTITYID_UNKNOWN,
            PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
        },
        locator::Locator,
        parameter::{Parameter, ParameterList, PID_USER_DATA},
        sequence::{SequenceNumber, SequenceNumberSet},
        time::Time,
    },
};

// ─── Helpers ────────────────────────────────────────────────────────────────

fn make_header() -> MessageHeader {
    MessageHeader {
        version: PROTOCOL_VERSION_2_3,
        vendor_id: VENDOR_ID_OXICTL,
        guid_prefix: GuidPrefix([0u8; 12]),
    }
}

fn make_message<'a>(subs: heapless::Vec<Submessage<'a>, 64>) -> Message<'a> {
    Message {
        header: make_header(),
        submessages: subs,
    }
}

fn round_trip<'buf>(msg: &Message<'_>, buf: &'buf mut [u8]) -> Message<'buf> {
    let n = serialize_message(msg, buf).expect("serialize_message");
    parse_message(&buf[..n]).expect("parse_message")
}

// ─── Group 1: Round-trip per submessage kind ────────────────────────────────

#[test]
fn round_trip_pad() {
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Pad).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    assert_eq!(parsed.submessages[0], Submessage::Pad);
}

#[test]
fn round_trip_data() {
    let payload = [0xDE, 0xAD, 0xBE, 0xEF];
    let data = Data {
        endianness: Endianness::Little,
        inline_qos_flag: false,
        data_flag: true,
        key_flag: false,
        non_standard_payload_flag: false,
        extra_flags: 0,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 1 },
        inline_qos: None,
        serialized_payload: &payload,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Data(data)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::Data(d) = &parsed.submessages[0] {
        assert!(d.data_flag);
        assert!(!d.inline_qos_flag);
        assert_eq!(d.writer_sn, SequenceNumber { high: 0, low: 1 });
        assert_eq!(d.serialized_payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    } else {
        panic!("expected Data submessage");
    }
}

#[test]
fn round_trip_data_frag() {
    let payload = [0x01, 0x02, 0x03, 0x04];
    let frag = DataFrag {
        endianness: Endianness::Little,
        inline_qos_flag: false,
        key_flag: false,
        non_standard_payload_flag: false,
        extra_flags: 0,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 5 },
        fragment_starting_num: FragmentNumber(1),
        fragments_in_submessage: 2,
        fragment_size: 512,
        sample_size: 1024,
        inline_qos: None,
        serialized_payload: &payload,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::DataFrag(frag)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::DataFrag(d) = &parsed.submessages[0] {
        assert_eq!(d.fragments_in_submessage, 2);
        assert_eq!(d.fragment_size, 512);
        assert_eq!(d.sample_size, 1024);
        assert_eq!(d.serialized_payload, &[0x01, 0x02, 0x03, 0x04]);
    } else {
        panic!("expected DataFrag submessage");
    }
}

#[test]
fn round_trip_heartbeat() {
    let hb = Heartbeat {
        endianness: Endianness::Little,
        final_flag: true,
        liveliness_flag: false,
        group_info_flag: false,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        first_sn: SequenceNumber { high: 0, low: 1 },
        last_sn: SequenceNumber { high: 0, low: 10 },
        count: 42,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Heartbeat(hb)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::Heartbeat(h) = &parsed.submessages[0] {
        assert!(h.final_flag);
        assert_eq!(h.count, 42);
        assert_eq!(h.first_sn, SequenceNumber { high: 0, low: 1 });
        assert_eq!(h.last_sn, SequenceNumber { high: 0, low: 10 });
    } else {
        panic!("expected Heartbeat submessage");
    }
}

#[test]
fn round_trip_heartbeat_frag() {
    let hbf = HeartbeatFrag {
        endianness: Endianness::Little,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 3 },
        last_fragment_num: FragmentNumber(7),
        count: 5,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::HeartbeatFrag(hbf)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::HeartbeatFrag(h) = &parsed.submessages[0] {
        assert_eq!(h.last_fragment_num, FragmentNumber(7));
        assert_eq!(h.count, 5);
    } else {
        panic!("expected HeartbeatFrag submessage");
    }
}

#[test]
fn round_trip_acknack() {
    let base = SequenceNumber { high: 0, low: 1 };
    let mut sn_set = SequenceNumberSet::empty(base);
    sn_set.set(SequenceNumber { high: 0, low: 1 }).unwrap();
    sn_set.set(SequenceNumber { high: 0, low: 3 }).unwrap();

    let an = AckNack {
        endianness: Endianness::Little,
        final_flag: true,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        reader_sn_state: sn_set,
        count: 1,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::AckNack(an)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::AckNack(a) = &parsed.submessages[0] {
        assert!(a.final_flag);
        assert_eq!(a.count, 1);
        assert!(a.reader_sn_state.is_set(SequenceNumber { high: 0, low: 1 }));
        assert!(a.reader_sn_state.is_set(SequenceNumber { high: 0, low: 3 }));
    } else {
        panic!("expected AckNack submessage");
    }
}

#[test]
fn round_trip_nack_frag() {
    let base = FragmentNumber(1);
    let mut fn_set = FragmentNumberSet::empty(base);
    fn_set.set(FragmentNumber(1)).unwrap();
    fn_set.set(FragmentNumber(4)).unwrap();

    let nf = NackFrag {
        endianness: Endianness::Little,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 2 },
        fragment_number_state: fn_set,
        count: 3,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::NackFrag(nf)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::NackFrag(n) = &parsed.submessages[0] {
        assert_eq!(n.count, 3);
        assert!(n.fragment_number_state.is_set(FragmentNumber(1)));
        assert!(n.fragment_number_state.is_set(FragmentNumber(4)));
    } else {
        panic!("expected NackFrag submessage");
    }
}

#[test]
fn round_trip_gap() {
    let base = SequenceNumber { high: 0, low: 5 };
    let gap_list = SequenceNumberSet::empty(base);
    let g = Gap {
        endianness: Endianness::Little,
        group_info_flag: false,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        gap_start: SequenceNumber { high: 0, low: 5 },
        gap_list,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Gap(g)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::Gap(gap) = &parsed.submessages[0] {
        assert_eq!(gap.gap_start, SequenceNumber { high: 0, low: 5 });
        assert_eq!(gap.gap_list.bitmap_base, base);
    } else {
        panic!("expected Gap submessage");
    }
}

#[test]
fn round_trip_info_timestamp() {
    let it = InfoTimestamp {
        endianness: Endianness::Little,
        invalidate_flag: false,
        timestamp: Some(Time {
            seconds: 1_000_000,
            fraction: 500,
        }),
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoTimestamp(it)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoTimestamp(t) = &parsed.submessages[0] {
        assert!(!t.invalidate_flag);
        assert_eq!(
            t.timestamp,
            Some(Time {
                seconds: 1_000_000,
                fraction: 500
            })
        );
    } else {
        panic!("expected InfoTimestamp submessage");
    }
}

#[test]
fn round_trip_info_timestamp_invalidate() {
    let it = InfoTimestamp {
        endianness: Endianness::Little,
        invalidate_flag: true,
        timestamp: None,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoTimestamp(it)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoTimestamp(t) = &parsed.submessages[0] {
        assert!(t.invalidate_flag);
        assert_eq!(t.timestamp, None);
    } else {
        panic!("expected InfoTimestamp submessage");
    }
}

#[test]
fn round_trip_info_source() {
    let is = InfoSource {
        endianness: Endianness::Little,
        protocol_version: PROTOCOL_VERSION_2_3,
        vendor_id: VENDOR_ID_OXICTL,
        guid_prefix: GuidPrefix([0xBB; 12]),
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoSource(is)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoSource(s) = &parsed.submessages[0] {
        assert_eq!(s.protocol_version, PROTOCOL_VERSION_2_3);
        assert_eq!(s.vendor_id, VENDOR_ID_OXICTL);
        assert_eq!(s.guid_prefix, GuidPrefix([0xBB; 12]));
    } else {
        panic!("expected InfoSource submessage");
    }
}

#[test]
fn round_trip_info_destination() {
    let id = InfoDestination {
        endianness: Endianness::Little,
        guid_prefix: GuidPrefix([0xCC; 12]),
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoDestination(id)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoDestination(d) = &parsed.submessages[0] {
        assert_eq!(d.guid_prefix, GuidPrefix([0xCC; 12]));
    } else {
        panic!("expected InfoDestination submessage");
    }
}

#[test]
fn round_trip_info_reply() {
    let loc1 = Locator::udp_v4(7400, [127, 0, 0, 1]);
    let loc2 = Locator::udp_v4(7401, [192, 168, 1, 1]);
    let mut unicast_list: heapless::Vec<Locator, 8> = heapless::Vec::new();
    unicast_list.push(loc1).unwrap();
    unicast_list.push(loc2).unwrap();
    let ir = InfoReply {
        endianness: Endianness::Little,
        multicast_flag: false,
        unicast_locator_list: unicast_list,
        multicast_locator_list: heapless::Vec::new(),
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoReply(ir)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 512];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoReply(r) = &parsed.submessages[0] {
        assert!(!r.multicast_flag);
        assert_eq!(r.unicast_locator_list.len(), 2);
        assert_eq!(r.unicast_locator_list[0], loc1);
        assert_eq!(r.unicast_locator_list[1], loc2);
    } else {
        panic!("expected InfoReply submessage");
    }
}

#[test]
fn round_trip_info_reply_ip4() {
    let uc = Locator::udp_v4(7400, [10, 0, 0, 1]);
    let ir4 = InfoReplyIp4 {
        endianness: Endianness::Little,
        multicast_flag: false,
        unicast_locator: uc,
        multicast_locator: None,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoReplyIp4(ir4)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoReplyIp4(r) = &parsed.submessages[0] {
        assert!(!r.multicast_flag);
        assert_eq!(r.unicast_locator, uc);
        assert_eq!(r.multicast_locator, None);
    } else {
        panic!("expected InfoReplyIp4 submessage");
    }
}

#[test]
fn round_trip_info_reply_ip4_multicast() {
    let uc = Locator::udp_v4(7400, [10, 0, 0, 1]);
    let mc = Locator::udp_v4(7401, [239, 255, 0, 1]);
    let ir4 = InfoReplyIp4 {
        endianness: Endianness::Little,
        multicast_flag: true,
        unicast_locator: uc,
        multicast_locator: Some(mc),
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::InfoReplyIp4(ir4)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    assert_eq!(parsed.submessages.len(), 1);
    if let Submessage::InfoReplyIp4(r) = &parsed.submessages[0] {
        assert!(r.multicast_flag);
        assert_eq!(r.unicast_locator, uc);
        assert_eq!(r.multicast_locator, Some(mc));
    } else {
        panic!("expected InfoReplyIp4 submessage");
    }
}

// ─── Group 2: Header validation ─────────────────────────────────────────────

#[test]
fn header_wrong_magic() {
    let mut bytes = [0u8; 24];
    bytes[0..4].copy_from_slice(b"XRTS");
    bytes[4] = 2;
    bytes[5] = 3;
    let result = parse_message(&bytes);
    assert_eq!(result, Err(RtpsError::InvalidMagic));
}

#[test]
fn header_version_1x() {
    let mut bytes = [0u8; 24];
    bytes[0..4].copy_from_slice(b"RTPS");
    bytes[4] = 1; // version major = 1
    bytes[5] = 0;
    let result = parse_message(&bytes);
    assert_eq!(result, Err(RtpsError::UnsupportedVersion));
}

#[test]
fn header_version_25_forward_compat() {
    // A future version 2.5 should be accepted as a compatible 2.x message
    let mut bytes = [0u8; 20];
    bytes[0..4].copy_from_slice(b"RTPS");
    bytes[4] = 2; // version major = 2
    bytes[5] = 5; // version minor = 5 (future)
                  // vendorId and guidPrefix can be zero
    let result = parse_message(&bytes);
    assert!(result.is_ok(), "version 2.5 should parse OK: {:?}", result);
    assert_eq!(result.unwrap().submessages.len(), 0);
}

// ─── Group 3: Truncated buffer errors ───────────────────────────────────────

#[test]
fn truncated_at_byte_5() {
    // Only 5 bytes — starts with RTPS but header is not complete
    let bytes = b"RTPs\x02"; // 5 bytes
    let result = parse_message(bytes);
    assert_eq!(result, Err(RtpsError::TruncatedHeader));
}

#[test]
fn truncated_at_byte_19() {
    // 19 bytes — one byte short of full header
    let mut bytes = [0u8; 19];
    bytes[0..4].copy_from_slice(b"RTPS");
    bytes[4] = 2;
    bytes[5] = 3;
    let result = parse_message(&bytes);
    assert_eq!(result, Err(RtpsError::TruncatedHeader));
}

#[test]
fn truncated_mid_submessage() {
    // Valid 20-byte header + submessage claiming 100 body bytes but only 4 bytes follow
    let mut bytes = [0u8; 24];
    bytes[0..4].copy_from_slice(b"RTPS");
    bytes[4] = 2;
    bytes[5] = 3;
    bytes[6] = 0x01; // vendorId
    bytes[7] = 0x10;
    // guidPrefix = zeros (bytes 8..20)
    // Submessage header at byte 20:
    bytes[20] = 0x07; // HEARTBEAT kind
    bytes[21] = 0x01; // flags: LE
    bytes[22] = 100; // octets_to_next_header = 100 (LE low byte)
    bytes[23] = 0;
    let result = parse_message(&bytes);
    assert_eq!(result, Err(RtpsError::BufferTooSmall));
}

// ─── Group 4: Unknown submessage skip ───────────────────────────────────────

#[test]
fn unknown_submessage_skipped() {
    // Build a valid RTPS header + unknown submessage (kind=0x77) + HEARTBEAT
    // The HEARTBEAT body is 28 bytes.
    let mut buf = [0u8; 512];

    // RTPS header (20 bytes)
    buf[0..4].copy_from_slice(b"RTPS");
    buf[4] = 2;
    buf[5] = 3;
    buf[6] = 0x01;
    buf[7] = 0x10;
    // guidPrefix = zeros

    // Unknown submessage: kind=0x77, flags=0x01 (LE), octets_to_next_header=4
    buf[20] = 0x77; // unknown kind
    buf[21] = 0x01; // LE
    buf[22] = 4; // 4 body bytes (LE)
    buf[23] = 0;
    // 4 padding bytes
    buf[24] = 0xAA;
    buf[25] = 0xBB;
    buf[26] = 0xCC;
    buf[27] = 0xDD;

    // HEARTBEAT submessage at offset 28: 4-byte header + 28-byte body = 32 bytes
    buf[28] = 0x07; // HEARTBEAT
    buf[29] = 0x01; // LE
    buf[30] = 28; // octets_to_next_header = 28 (body)
    buf[31] = 0;
    // Body: reader_id (4) + writer_id (4) + first_sn (8) + last_sn (8) + count (4)
    // All zeros except last_sn.low = 5, count = 1
    // reader_id = ENTITYID_UNKNOWN = [0,0,0,0]
    buf[32..36].copy_from_slice(&[0, 0, 0, 0]);
    // writer_id = ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER = [0,0x01,0x00,0xC2]
    buf[36..40].copy_from_slice(&[0, 0x01, 0x00, 0xC2]);
    // first_sn = {high:0, low:1} LE
    buf[40..44].copy_from_slice(&[0, 0, 0, 0]); // high i32
    buf[44..48].copy_from_slice(&[1, 0, 0, 0]); // low u32
                                                // last_sn = {high:0, low:5} LE
    buf[48..52].copy_from_slice(&[0, 0, 0, 0]); // high
    buf[52..56].copy_from_slice(&[5, 0, 0, 0]); // low
                                                // count = 1 LE
    buf[56..60].copy_from_slice(&[1, 0, 0, 0]);

    let result = parse_message(&buf[..60]);
    assert!(result.is_ok(), "should parse successfully: {:?}", result);
    let msg = result.unwrap();
    assert_eq!(
        msg.submessages.len(),
        1,
        "should have exactly 1 submessage (the HEARTBEAT), unknown skipped"
    );
    assert!(
        matches!(msg.submessages[0], Submessage::Heartbeat(_)),
        "submessage should be Heartbeat"
    );
    if let Submessage::Heartbeat(h) = &msg.submessages[0] {
        assert_eq!(h.last_sn, SequenceNumber { high: 0, low: 5 });
    }
}

// ─── Group 5: Endianness round-trips ────────────────────────────────────────

#[test]
fn data_round_trip_big_endian() {
    let payload = [0x11, 0x22, 0x33, 0x44];
    let data = Data {
        endianness: Endianness::Big,
        inline_qos_flag: false,
        data_flag: true,
        key_flag: false,
        non_standard_payload_flag: false,
        extra_flags: 0,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 100 },
        inline_qos: None,
        serialized_payload: &payload,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Data(data)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    if let Submessage::Data(d) = &parsed.submessages[0] {
        assert_eq!(d.endianness, Endianness::Big);
        assert_eq!(d.writer_sn, SequenceNumber { high: 0, low: 100 });
        assert_eq!(d.serialized_payload, &[0x11, 0x22, 0x33, 0x44]);
    } else {
        panic!("expected Data submessage");
    }
}

#[test]
fn data_round_trip_little_endian() {
    let payload = [0x55, 0x66, 0x77, 0x88];
    let data = Data {
        endianness: Endianness::Little,
        inline_qos_flag: false,
        data_flag: true,
        key_flag: false,
        non_standard_payload_flag: false,
        extra_flags: 0,
        reader_id: ENTITYID_UNKNOWN,
        writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        writer_sn: SequenceNumber { high: 0, low: 200 },
        inline_qos: None,
        serialized_payload: &payload,
    };
    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
    subs.push(Submessage::Data(data)).unwrap();
    let msg = make_message(subs);
    let mut buf = [0u8; 256];
    let parsed = round_trip(&msg, &mut buf);
    if let Submessage::Data(d) = &parsed.submessages[0] {
        assert_eq!(d.endianness, Endianness::Little);
        assert_eq!(d.writer_sn, SequenceNumber { high: 0, low: 200 });
        assert_eq!(d.serialized_payload, &[0x55, 0x66, 0x77, 0x88]);
    } else {
        panic!("expected Data submessage");
    }
}

// ─── Group 6: SequenceNumberSet bitmap ──────────────────────────────────────

#[test]
fn seqnumset_empty() {
    let base = SequenceNumber { high: 0, low: 10 };
    let s = SequenceNumberSet::empty(base);
    assert_eq!(s.serialized_len(), 12); // 8 (base) + 4 (num_bits=0) + 0 words
    assert!(!s.is_set(SequenceNumber { high: 0, low: 10 }));
    assert_eq!(s.iter().count(), 0);
}

#[test]
fn seqnumset_dense_256() {
    let base = SequenceNumber { high: 0, low: 1 };
    let mut s = SequenceNumberSet::empty(base);
    // Set bit at offset 0 (sn=1) and offset 255 (sn=256)
    s.set(SequenceNumber { high: 0, low: 1 }).unwrap();
    s.set(SequenceNumber { high: 0, low: 256 }).unwrap();
    assert!(s.is_set(SequenceNumber { high: 0, low: 1 }));
    assert!(s.is_set(SequenceNumber { high: 0, low: 256 }));
    assert!(!s.is_set(SequenceNumber { high: 0, low: 2 }));
    let collected: heapless::Vec<SequenceNumber, 16> = s.iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn seqnumset_sparse() {
    let base = SequenceNumber { high: 0, low: 0 };
    let mut s = SequenceNumberSet::empty(base);
    s.set(SequenceNumber { high: 0, low: 0 }).unwrap();
    s.set(SequenceNumber { high: 0, low: 100 }).unwrap();
    s.set(SequenceNumber { high: 0, low: 200 }).unwrap();

    // Round-trip via serialize/parse
    let mut body_buf = [0u8; 128];
    let mut w =
        crate::protocol::dds::byte_cursor::ByteWriter::new(&mut body_buf, Endianness::Little);
    s.serialize(&mut w).unwrap();
    let written = w.position();
    let mut cur = crate::protocol::dds::byte_cursor::ByteCursor::new(
        &body_buf[..written],
        Endianness::Little,
    );
    let parsed = SequenceNumberSet::parse(&mut cur).unwrap();
    assert!(parsed.is_set(SequenceNumber { high: 0, low: 0 }));
    assert!(parsed.is_set(SequenceNumber { high: 0, low: 100 }));
    assert!(parsed.is_set(SequenceNumber { high: 0, low: 200 }));
}

// ─── Group 7: ParameterList sentinel handling ────────────────────────────────

#[test]
fn param_list_missing_sentinel() {
    // One valid parameter but then truncated — no PID_SENTINEL
    // PID=0x002C (PID_USER_DATA), length=4, value=[1,2,3,4]  — then EOF
    let bytes: [u8; 8] = [
        0x2C, 0x00, // PID_USER_DATA LE
        0x04, 0x00, // length = 4 LE
        0x01, 0x02, 0x03, 0x04, // value
              // NO sentinel — truncated
    ];
    let mut cur = crate::protocol::dds::byte_cursor::ByteCursor::new(&bytes, Endianness::Little);
    let result = ParameterList::parse(&mut cur);
    assert!(result.is_err(), "missing sentinel should return Err");
}

#[test]
fn param_list_five_params() {
    // Build a ParameterList with 5 parameters, serialize, parse back, verify count
    let values: &[&[u8]] = &[
        &[0x01u8, 0x00, 0x00, 0x00],
        &[0x02u8, 0x00, 0x00, 0x00],
        &[0x03u8, 0x00, 0x00, 0x00],
        &[0x04u8, 0x00, 0x00, 0x00],
        &[0x05u8, 0x00, 0x00, 0x00],
    ];
    let mut pl = ParameterList::new();
    for (i, v) in values.iter().enumerate() {
        pl.push(Parameter {
            pid: PID_USER_DATA + i as u16,
            value: v,
        })
        .unwrap();
    }
    assert_eq!(pl.len(), 5);

    let mut buf = [0u8; 256];
    let mut w = crate::protocol::dds::byte_cursor::ByteWriter::new(&mut buf, Endianness::Little);
    pl.serialize(&mut w).unwrap();
    let written = w.position();

    let mut cur =
        crate::protocol::dds::byte_cursor::ByteCursor::new(&buf[..written], Endianness::Little);
    let parsed = ParameterList::parse(&mut cur).unwrap();
    assert_eq!(parsed.len(), 5);
    for (i, p) in parsed.iter().enumerate() {
        assert_eq!(p.pid, PID_USER_DATA + i as u16);
    }
}

// ─── Group 8: Structural fixture (SPDP DATA) ────────────────────────────────

#[test]
fn structural_fixture_spdp_data() {
    // Hand-crafted RTPS 2.3 message: 20-byte header + 1 DATA submessage (4-byte header + 24-byte body)
    // Total = 48 bytes
    #[rustfmt::skip]
    let bytes: [u8; 48] = [
        // RTPS header (20 bytes)
        0x52, 0x54, 0x50, 0x53, // "RTPS" magic
        0x02, 0x03,             // version 2.3
        0x01, 0x10,             // vendorId = VENDOR_ID_OXICTL
        0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, // guidPrefix (12 bytes)
        0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,

        // DATA submessage header (4 bytes)
        0x15,       // kind = DATA (0x15)
        0x05,       // flags: E=1 (LE), D=1 (data present) → 0b0000_0101
        0x18, 0x00, // octets_to_next_header = 24 (LE)

        // DATA submessage body (24 bytes)
        0x00, 0x00, // extra_flags = 0
        0x10, 0x00, // octets_to_inline_qos = 16 (LE)
        0x00, 0x00, 0x00, 0x00, // readerId = ENTITYID_UNKNOWN
        0x00, 0x01, 0x00, 0xC2, // writerId = ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER
        0x00, 0x00, 0x00, 0x00, // writerSN.high = 0 (i32 LE)
        0x01, 0x00, 0x00, 0x00, // writerSN.low = 1 (u32 LE)
        0xDE, 0xAD, 0xBE, 0xEF, // serialized_payload (4 bytes)
    ];

    let msg = parse_message(&bytes).expect("fixture parse failed");

    assert_eq!(msg.header.version, PROTOCOL_VERSION_2_3);
    assert_eq!(msg.header.vendor_id, VENDOR_ID_OXICTL);
    assert_eq!(msg.header.guid_prefix, GuidPrefix([0xAA; 12]));
    assert_eq!(msg.submessages.len(), 1);

    if let Submessage::Data(d) = &msg.submessages[0] {
        assert_eq!(d.endianness, Endianness::Little);
        assert!(d.data_flag);
        assert!(!d.inline_qos_flag);
        assert_eq!(d.reader_id, ENTITYID_UNKNOWN);
        assert_eq!(
            d.writer_id,
            EntityId {
                entity_key: [0, 0x01, 0x00],
                entity_kind: 0xC2
            }
        );
        assert_eq!(d.writer_id, ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER);
        assert_eq!(d.writer_sn, SequenceNumber { high: 0, low: 1 });
        assert_eq!(d.serialized_payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    } else {
        panic!("expected Submessage::Data, got {:?}", &msg.submessages[0]);
    }
}
