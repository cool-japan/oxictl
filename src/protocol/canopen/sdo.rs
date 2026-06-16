//! CANopen SDO (Service Data Object) — acyclic object dictionary access.
//!
//! SDOs use a client-server model for configuration and parameter access.
//! Two transfer modes:
//!   - Expedited: ≤4 bytes in a single frame
//!   - Segmented: multiple frames for larger data

use super::object_dict::{OdEntryValue, OdError, StaticOd};
use super::pdo::CanFrame;

/// SDO command specifiers.
#[repr(u8)]
pub enum SdoCs {
    /// Initiate upload (read request from client).
    UploadInitiateReq = 0x40,
    /// Upload initiate response.
    UploadInitiateResp = 0x42,
    /// Initiate download (write request from client).
    DownloadInitiateReq = 0x22,
    /// Download initiate response.
    DownloadInitiateResp = 0x60,
    /// Abort transfer.
    Abort = 0x80,
}

// ─── SDO Abort Codes (CiA 301 §7.2.4.3.17) ──────────────────────────────────

/// Standard SDO abort codes per CiA 301.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SdoAbortCode {
    /// Toggle bit not alternated.
    ToggleBitNotAlternated = 0x0503_0000,
    /// SDO protocol timed out.
    SdoProtocolTimedOut = 0x0504_0000,
    /// Client/server command specifier not valid or unknown.
    InvalidCommandSpecifier = 0x0504_0001,
    /// Object does not exist in the object dictionary.
    ObjectDoesNotExist = 0x0602_0000,
    /// Object cannot be mapped to the PDO.
    ObjectCannotBeMapped = 0x0604_0041,
    /// PDO length exceeded.
    PdoLengthExceeded = 0x0604_0042,
    /// Sub-index does not exist.
    SubindexDoesNotExist = 0x0609_0011,
    /// Value range of parameter exceeded.
    ValueRangeExceeded = 0x0609_0030,
    /// Attempt to write to a read-only object.
    ReadOnly = 0x0603_0000,
    /// Attempt to read from a write-only object.
    WriteOnly = 0x0603_0100,
    /// Data type mismatch.
    DataTypeMismatch = 0x0607_0010,
    /// General error.
    GeneralError = 0x0800_0000,
    /// Data cannot be transferred or stored to the application.
    DataTransferError = 0x0800_0020,
}

impl SdoAbortCode {
    /// Convert to the 4-byte little-endian wire representation.
    pub fn to_le_bytes(self) -> [u8; 4] {
        (self as u32).to_le_bytes()
    }

    /// Try to decode from a u32 raw value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0x0503_0000 => Some(Self::ToggleBitNotAlternated),
            0x0504_0000 => Some(Self::SdoProtocolTimedOut),
            0x0504_0001 => Some(Self::InvalidCommandSpecifier),
            0x0602_0000 => Some(Self::ObjectDoesNotExist),
            0x0604_0041 => Some(Self::ObjectCannotBeMapped),
            0x0604_0042 => Some(Self::PdoLengthExceeded),
            0x0609_0011 => Some(Self::SubindexDoesNotExist),
            0x0609_0030 => Some(Self::ValueRangeExceeded),
            0x0603_0000 => Some(Self::ReadOnly),
            0x0603_0100 => Some(Self::WriteOnly),
            0x0607_0010 => Some(Self::DataTypeMismatch),
            0x0800_0000 => Some(Self::GeneralError),
            0x0800_0020 => Some(Self::DataTransferError),
            _ => None,
        }
    }
}

// ─── SdoError ─────────────────────────────────────────────────────────────────

/// Error type returned by `SdoServer::process`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdoError {
    /// Unrecognised command specifier.
    UnknownCommandSpecifier(u8),
    /// The OD access was rejected with an abort code.
    Abort(SdoAbortCode),
}

// ─── OdEntry value → expedited SDO byte helpers ──────────────────────────────

/// Map `OdError` to the appropriate `SdoAbortCode`.
fn od_error_to_abort(err: OdError) -> SdoAbortCode {
    match err {
        OdError::NotFound | OdError::IndexNotFound => SdoAbortCode::ObjectDoesNotExist,
        OdError::SubindexNotFound => SdoAbortCode::SubindexDoesNotExist,
        OdError::ReadOnly => SdoAbortCode::ReadOnly,
        OdError::WriteOnly => SdoAbortCode::WriteOnly,
        OdError::TypeMismatch => SdoAbortCode::DataTypeMismatch,
        OdError::Full | OdError::AccessDenied => SdoAbortCode::GeneralError,
    }
}

/// Encode an `OdEntryValue` into the 4-byte expedited SDO data field.
///
/// Returns `(data_bytes, n_empty)` where `n_empty` is the number of bytes
/// in the 4-byte field that do NOT carry data (used in the cs specifier).
fn encode_expedited(value: OdEntryValue) -> ([u8; 4], u8) {
    let raw = value.to_le_bytes();
    let sz = value.byte_size().min(4);
    let mut data = [0u8; 4];
    data[..sz].copy_from_slice(&raw[..sz]);
    let n_empty = 4u8.saturating_sub(sz as u8);
    (data, n_empty)
}

/// Decode the 4-byte expedited SDO data field into an `OdEntryValue` whose
/// type matches the existing entry in the OD.
///
/// `n_empty` comes from the cs byte: `(cs >> 2) & 0x03`.
fn decode_expedited(existing: OdEntryValue, data: [u8; 4]) -> Result<OdEntryValue, SdoAbortCode> {
    match existing {
        OdEntryValue::U8(_) => Ok(OdEntryValue::U8(data[0])),
        OdEntryValue::I8(_) => Ok(OdEntryValue::I8(data[0] as i8)),
        OdEntryValue::U16(_) => Ok(OdEntryValue::U16(u16::from_le_bytes([data[0], data[1]]))),
        OdEntryValue::I16(_) => Ok(OdEntryValue::I16(i16::from_le_bytes([data[0], data[1]]))),
        OdEntryValue::U32(_) => Ok(OdEntryValue::U32(u32::from_le_bytes([
            data[0], data[1], data[2], data[3],
        ]))),
        OdEntryValue::I32(_) => Ok(OdEntryValue::I32(i32::from_le_bytes([
            data[0], data[1], data[2], data[3],
        ]))),
        OdEntryValue::Bool(_) => Ok(OdEntryValue::Bool(data[0] != 0)),
        OdEntryValue::OctetString(_) => {
            // Expedited transfer cannot carry 8 bytes; we treat the 4 data bytes
            // as the first 4 bytes of the octet string (zero-padded).
            let mut s = [0u8; 8];
            s[..4].copy_from_slice(&data);
            Ok(OdEntryValue::OctetString(s))
        }
    }
}

// ─── SdoServer ────────────────────────────────────────────────────────────────

/// Expedited-only SDO server that operates on a `StaticOd<N>`.
///
/// `SdoServer` processes an incoming SDO client request CAN frame and
/// produces a response CAN frame (or an abort frame on error).
///
/// Only expedited (≤4 byte) uploads and downloads are supported; segmented
/// transfers are handled separately by `SdoSegmentTransfer`.
///
/// # COB-IDs
///
/// Per CiA 301, for node `node_id`:
/// - Server receives requests on COB-ID `0x600 + node_id`
/// - Server sends responses on COB-ID `0x580 + node_id`
pub struct SdoServer<const N: usize> {
    node_id: u8,
    /// Receive COB-ID (0x600 + node_id).
    rx_cob_id: u32,
    /// Transmit COB-ID (0x580 + node_id).
    tx_cob_id: u32,
}

impl<const N: usize> SdoServer<N> {
    /// Create a new `SdoServer` for the given `node_id` (1–127).
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            rx_cob_id: 0x600 + node_id as u32,
            tx_cob_id: 0x580 + node_id as u32,
        }
    }

    /// Node ID this server belongs to.
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Expected receive COB-ID.
    pub fn rx_cob_id(&self) -> u32 {
        self.rx_cob_id
    }

    /// Transmit COB-ID used in responses.
    pub fn tx_cob_id(&self) -> u32 {
        self.tx_cob_id
    }

    /// Process an incoming SDO request frame and produce a response frame.
    ///
    /// The `od` parameter is the object dictionary that will be read from or
    /// written to.  The response frame is always 8 bytes (DLC=8).
    ///
    /// # Errors
    ///
    /// Returns `Err(SdoError::UnknownCommandSpecifier(cs))` if the command
    /// specifier byte is not recognised.  The caller should send an abort
    /// frame in this case (use `build_abort_frame`).
    ///
    /// Returns `Err(SdoError::Abort(code))` if the OD access fails; the
    /// returned `CanFrame` is the abort frame to transmit.
    pub fn process(
        &self,
        req_frame: &CanFrame,
        od: &mut StaticOd<N>,
    ) -> Result<CanFrame, SdoError> {
        let data = &req_frame.data;
        let cs = data[0];
        let index = u16::from_le_bytes([data[1], data[2]]);
        let subindex = data[3];
        let req_data = [data[4], data[5], data[6], data[7]];

        // cs dispatch
        match cs & 0xE0 {
            // Upload initiate request: cs = 0x40
            0x40 => self.handle_upload(od, index, subindex),
            // Download initiate request: cs = 0x20 (expedited bit may vary)
            0x20 => self.handle_download(od, index, subindex, cs, req_data),
            0x80 => {
                // Client abort — no response needed; return the same abort frame
                Ok(CanFrame::new(self.tx_cob_id, &req_frame.data))
            }
            _ => Err(SdoError::UnknownCommandSpecifier(cs)),
        }
    }

    fn handle_upload(
        &self,
        od: &StaticOd<N>,
        index: u16,
        subindex: u8,
    ) -> Result<CanFrame, SdoError> {
        let value = od
            .read(index, subindex)
            .map_err(|e| SdoError::Abort(od_error_to_abort(e)))?;
        let (val_data, n_empty) = encode_expedited(value);
        // cs = 0x43 | (n_empty << 2) — expedited, size indicated
        let resp_cs = 0x43u8 | (n_empty << 2);
        let idx = index.to_le_bytes();
        let resp_data = [
            resp_cs,
            idx[0],
            idx[1],
            subindex,
            val_data[0],
            val_data[1],
            val_data[2],
            val_data[3],
        ];
        Ok(CanFrame::new(self.tx_cob_id, &resp_data))
    }

    fn handle_download(
        &self,
        od: &mut StaticOd<N>,
        index: u16,
        subindex: u8,
        _cs: u8,
        req_data: [u8; 4],
    ) -> Result<CanFrame, SdoError> {
        // Peek at the existing value to know the target type.
        let existing = od
            .read(index, subindex)
            .map_err(|e| SdoError::Abort(od_error_to_abort(e)))?;

        // Decode the incoming bytes as the target type.
        let new_val = decode_expedited(existing, req_data).map_err(SdoError::Abort)?;

        od.write(index, subindex, new_val)
            .map_err(|e| SdoError::Abort(od_error_to_abort(e)))?;

        // Respond with Download Initiate Response (cs = 0x60).
        let idx = index.to_le_bytes();
        let resp_data = [0x60u8, idx[0], idx[1], subindex, 0, 0, 0, 0];
        Ok(CanFrame::new(self.tx_cob_id, &resp_data))
    }

    /// Build an SDO abort frame for the given abort code.
    ///
    /// The abort frame is sent on the server's transmit COB-ID and follows
    /// the CiA 301 layout: `[0x80, idx_lo, idx_hi, sub, abort[0..4]]`.
    pub fn build_abort_frame(&self, index: u16, subindex: u8, code: SdoAbortCode) -> CanFrame {
        let idx = index.to_le_bytes();
        let ac = (code as u32).to_le_bytes();
        CanFrame::new(
            self.tx_cob_id,
            &[0x80, idx[0], idx[1], subindex, ac[0], ac[1], ac[2], ac[3]],
        )
    }
}

/// SDO frame (8 bytes: cs + index_lo + index_hi + sub_index + data[4]).
#[derive(Debug, Clone, Copy)]
pub struct SdoFrame {
    pub cs: u8,
    pub index: u16,
    pub sub_index: u8,
    pub data: [u8; 4],
}

impl SdoFrame {
    /// Create upload request (read) frame.
    pub fn upload_req(index: u16, sub_index: u8) -> Self {
        Self {
            cs: SdoCs::UploadInitiateReq as u8,
            index,
            sub_index,
            data: [0u8; 4],
        }
    }

    /// Create download request (write) frame with u32 data.
    pub fn download_req(index: u16, sub_index: u8, data: u32) -> Self {
        Self {
            cs: SdoCs::DownloadInitiateReq as u8,
            index,
            sub_index,
            data: data.to_le_bytes(),
        }
    }

    /// Serialize to 8-byte CAN frame data.
    pub fn to_bytes(&self) -> [u8; 8] {
        let idx = self.index.to_le_bytes();
        [
            self.cs,
            idx[0],
            idx[1],
            self.sub_index,
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
        ]
    }

    /// Parse from 8-byte CAN frame data.
    pub fn from_bytes(b: &[u8; 8]) -> Self {
        Self {
            cs: b[0],
            index: u16::from_le_bytes([b[1], b[2]]),
            sub_index: b[3],
            data: [b[4], b[5], b[6], b[7]],
        }
    }

    /// Extract u32 from data field.
    pub fn data_u32(&self) -> u32 {
        u32::from_le_bytes(self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::super::object_dict::{AccessType, DataType, OdEntry, OdEntryValue, StaticOd};
    use super::*;

    #[test]
    fn sdo_frame_roundtrip() {
        let frame = SdoFrame::download_req(0x6040, 0, 0x0006);
        let bytes = frame.to_bytes();
        let parsed = SdoFrame::from_bytes(&bytes);
        assert_eq!(parsed.index, 0x6040);
        assert_eq!(parsed.sub_index, 0);
        assert_eq!(parsed.data_u32(), 0x0006);
    }

    #[test]
    fn sdo_upload_req_cs() {
        let frame = SdoFrame::upload_req(0x1000, 0);
        assert_eq!(frame.cs, SdoCs::UploadInitiateReq as u8);
    }

    // ── SdoServer tests ──────────────────────────────────────────────────────

    fn make_od() -> StaticOd<32> {
        let mut od = StaticOd::<32>::new();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RO,
            OdEntryValue::U32(0x0402),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x6040,
            0,
            DataType::Unsigned16,
            AccessType::RW,
            OdEntryValue::U16(0),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x6041,
            0,
            DataType::Unsigned16,
            AccessType::RO,
            OdEntryValue::U16(0x0237),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x2100,
            0,
            DataType::Unsigned8,
            AccessType::WO,
            OdEntryValue::U8(0),
        ))
        .unwrap();
        od
    }

    fn make_upload_req(index: u16, subindex: u8) -> CanFrame {
        let idx = index.to_le_bytes();
        CanFrame::new(0x601, &[0x40, idx[0], idx[1], subindex, 0, 0, 0, 0])
    }

    fn make_download_req(index: u16, subindex: u8, val: u32) -> CanFrame {
        let idx = index.to_le_bytes();
        let d = val.to_le_bytes();
        // cs = 0x23: download initiate, expedited, size indicated, n=0
        CanFrame::new(
            0x601,
            &[0x23, idx[0], idx[1], subindex, d[0], d[1], d[2], d[3]],
        )
    }

    #[test]
    fn sdo_server_upload_u32() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_upload_req(0x1000, 0);
        let resp = server.process(&req, &mut od).unwrap();
        assert_eq!(resp.cob_id, 0x581);
        // Response cs upper bits = 0x40, expedited, size indicated
        assert_eq!(resp.data[0] & 0xE0, 0x40);
        let val = u32::from_le_bytes([resp.data[4], resp.data[5], resp.data[6], resp.data[7]]);
        assert_eq!(val, 0x0402);
    }

    #[test]
    fn sdo_server_upload_u16() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_upload_req(0x6041, 0);
        let resp = server.process(&req, &mut od).unwrap();
        let val = u16::from_le_bytes([resp.data[4], resp.data[5]]);
        assert_eq!(val, 0x0237);
    }

    #[test]
    fn sdo_server_download_u16() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_download_req(0x6040, 0, 0x000F);
        let resp = server.process(&req, &mut od).unwrap();
        // Download initiate response cs = 0x60
        assert_eq!(resp.data[0], 0x60);
        // Verify OD was updated
        let val = od.read(0x6040, 0).unwrap();
        assert_eq!(val, OdEntryValue::U16(0x000F));
    }

    #[test]
    fn sdo_server_upload_write_only_returns_abort() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_upload_req(0x2100, 0);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::WriteOnly));
    }

    #[test]
    fn sdo_server_download_read_only_returns_abort() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_download_req(0x1000, 0, 0xFFFF);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::ReadOnly));
    }

    #[test]
    fn sdo_server_object_does_not_exist() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        let req = make_upload_req(0x9999, 0);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::ObjectDoesNotExist));
    }

    #[test]
    fn upload_subindex_does_not_exist() {
        let mut od = StaticOd::<32>::new();
        od.insert(OdEntry::new(
            0x1018,
            0,
            DataType::Unsigned8,
            AccessType::RO,
            OdEntryValue::U8(4),
        ))
        .unwrap();
        let server = SdoServer::<32>::new(1);
        // Subindex 5 does not exist under 0x1018, but the index itself does.
        let req = make_upload_req(0x1018, 5);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::SubindexDoesNotExist));
    }

    #[test]
    fn upload_object_does_not_exist() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        // Index 0x8888 is not present in the OD at all.
        let req = make_upload_req(0x8888, 0);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::ObjectDoesNotExist));
    }

    #[test]
    fn upload_write_only_returns_error() {
        let mut od = make_od();
        let server = SdoServer::<32>::new(1);
        // 0x2100:0 is AccessType::WO — upload must be rejected.
        let req = make_upload_req(0x2100, 0);
        let err = server.process(&req, &mut od).unwrap_err();
        assert_eq!(err, SdoError::Abort(SdoAbortCode::WriteOnly));
    }

    #[test]
    fn sdo_abort_code_round_trip() {
        let code = SdoAbortCode::SubindexDoesNotExist;
        let bytes = code.to_le_bytes();
        let v = u32::from_le_bytes(bytes);
        assert_eq!(SdoAbortCode::from_u32(v), Some(code));
    }

    #[test]
    fn sdo_server_build_abort_frame() {
        let server = SdoServer::<32>::new(3);
        let frame = server.build_abort_frame(0x6040, 0, SdoAbortCode::ReadOnly);
        assert_eq!(frame.cob_id, 0x583);
        assert_eq!(frame.data[0], 0x80);
        let ac = u32::from_le_bytes([frame.data[4], frame.data[5], frame.data[6], frame.data[7]]);
        assert_eq!(ac, SdoAbortCode::ReadOnly as u32);
    }

    #[test]
    fn sdo_server_cob_ids() {
        let server = SdoServer::<32>::new(5);
        assert_eq!(server.rx_cob_id(), 0x605);
        assert_eq!(server.tx_cob_id(), 0x585);
        assert_eq!(server.node_id(), 5);
    }
}
