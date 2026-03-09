//! CANopen SDO Segmented Transfer protocol.
//!
//! Implements download (client→server) and upload (server→client)
//! segmented SDO transfers for data larger than 4 bytes.
//! Toggle bit is tracked per CiA 301 specification.

use heapless::Vec;

/// SDO abort codes (CiA 301, section 7.2.4.3.17).
pub const ABORT_TOGGLE_BIT: u32 = 0x0503_0000;
pub const ABORT_TIMEOUT: u32 = 0x0504_0000;
pub const ABORT_INVALID_CS: u32 = 0x0504_0001;
pub const ABORT_OBJ_NOT_EXIST: u32 = 0x0602_0000;
pub const ABORT_SUBIDX_NOT_EXIST: u32 = 0x0609_0011;
pub const ABORT_VALUE_RANGE: u32 = 0x0609_0030;
pub const ABORT_GENERAL: u32 = 0x0800_0000;
pub const ABORT_DATA_TRANSFER: u32 = 0x0800_0020;

/// SDO command specifier constants for segmented protocol.
/// Per CiA 301, the CS byte is composed of bitfields rather than unique values.
pub mod sdo_cs {
    /// Download initiate request base (client, cs = 0x21 | size_bits).
    pub const DOWNLOAD_INIT_REQ: u8 = 0x21;
    /// Download initiate response (server).
    pub const DOWNLOAD_INIT_RESP: u8 = 0x60;
    /// Download segment request base (client, cs = toggle<<4 | n<<1 | last).
    pub const DOWNLOAD_SEG_REQ_BASE: u8 = 0x00;
    /// Download segment response base (server, cs = 0x20 | toggle<<4).
    pub const DOWNLOAD_SEG_RESP_BASE: u8 = 0x20;
    /// Upload initiate request (client).
    pub const UPLOAD_INIT_REQ: u8 = 0x40;
    /// Upload initiate response (server).
    pub const UPLOAD_INIT_RESP: u8 = 0x41;
    /// Upload segment request base (client, cs = 0x60 | toggle<<4).
    pub const UPLOAD_SEG_REQ_BASE: u8 = 0x60;
    /// Upload segment response base (server, cs = toggle<<4 | n<<1 | last).
    pub const UPLOAD_SEG_RESP_BASE: u8 = 0x00;
    /// Abort transfer.
    pub const ABORT: u8 = 0x80;
}

/// Transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDir {
    Download,
    Upload,
}

/// State of a segmented SDO transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdoSegState {
    Idle,
    Initiated,
    InProgress,
    Complete,
    Aborted,
}

/// Segmented SDO transfer handler.
///
/// `BUF` is the maximum number of bytes that can be transferred.
pub struct SdoSegmentTransfer<const BUF: usize> {
    state: SdoSegState,
    direction: TransferDir,
    index: u16,
    sub_index: u8,
    /// Accumulated data buffer.
    data: Vec<u8, BUF>,
    /// Total expected length (from initiate frame).
    total_len: usize,
    /// Current toggle bit (false=0, true=1).
    toggle: bool,
    /// Number of segments transferred.
    segment_count: u32,
    /// Abort code if aborted.
    abort_code: u32,
}

impl<const BUF: usize> SdoSegmentTransfer<BUF> {
    /// Create a new segmented transfer handler in idle state.
    pub fn new() -> Self {
        Self {
            state: SdoSegState::Idle,
            direction: TransferDir::Download,
            index: 0,
            sub_index: 0,
            data: Vec::new(),
            total_len: 0,
            toggle: false,
            segment_count: 0,
            abort_code: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> SdoSegState {
        self.state
    }

    /// Number of segments transferred.
    pub fn segment_count(&self) -> u32 {
        self.segment_count
    }

    /// Accumulated data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Abort code (valid if state == Aborted).
    pub fn abort_code(&self) -> u32 {
        self.abort_code
    }

    // -----------------------------------------------------------------------
    // Download (client → server): receive data from client
    // -----------------------------------------------------------------------

    /// Process a download initiate request from client.
    /// Returns the server's initiate response frame.
    pub fn download_initiate(&mut self, frame: &[u8; 8]) -> Result<[u8; 8], u32> {
        // frame[0] = cs byte, check it's a download initiate
        // Expedited bit (bit 1) must be 0 for segmented, or size indicator
        let cs = frame[0];
        if cs & 0xE0 != 0x20 {
            return Err(ABORT_INVALID_CS);
        }
        self.index = u16::from_le_bytes([frame[1], frame[2]]);
        self.sub_index = frame[3];
        // Size indicator (bit 0 of cs): if set, bytes 4-7 contain total size
        if cs & 0x01 != 0 {
            self.total_len = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
        } else {
            self.total_len = 0;
        }
        self.data.clear();
        self.toggle = false;
        self.segment_count = 0;
        self.direction = TransferDir::Download;
        self.state = SdoSegState::Initiated;

        // Build response: cs=0x60, index, sub, 0,0,0,0
        let idx = self.index.to_le_bytes();
        Ok([0x60, idx[0], idx[1], self.sub_index, 0, 0, 0, 0])
    }

    /// Process a download segment frame from client.
    /// Returns the server's segment response frame.
    pub fn download_segment(&mut self, frame: &[u8; 8]) -> Result<[u8; 8], u32> {
        if self.state != SdoSegState::Initiated && self.state != SdoSegState::InProgress {
            return Err(ABORT_GENERAL);
        }
        let cs = frame[0];
        // Toggle bit is bit 4 of cs
        let frame_toggle = (cs & 0x10) != 0;
        if self.segment_count > 0 && frame_toggle == self.toggle {
            // Toggle bit not alternated
            self.state = SdoSegState::Aborted;
            self.abort_code = ABORT_TOGGLE_BIT;
            return Err(ABORT_TOGGLE_BIT);
        }
        self.toggle = frame_toggle;

        // Number of bytes that do NOT contain data in segment (bits 3:1)
        let n = ((cs >> 1) & 0x07) as usize;
        let data_len = 7usize.saturating_sub(n);
        let last = (cs & 0x01) != 0;

        for i in 0..data_len {
            if self.data.push(frame[1 + i]).is_err() {
                self.state = SdoSegState::Aborted;
                self.abort_code = ABORT_DATA_TRANSFER;
                return Err(ABORT_DATA_TRANSFER);
            }
        }
        self.segment_count += 1;
        self.state = if last {
            SdoSegState::Complete
        } else {
            SdoSegState::InProgress
        };

        // Response: cs = 0x20 | (toggle << 4)
        let resp_cs = 0x20u8 | (if frame_toggle { 0x10 } else { 0x00 });
        Ok([resp_cs, 0, 0, 0, 0, 0, 0, 0])
    }

    // -----------------------------------------------------------------------
    // Upload (server → client): send data to client
    // -----------------------------------------------------------------------

    /// Initiate an upload transfer. Prepares internal buffer with data to send.
    /// Returns the upload initiate response frame to send to client.
    pub fn upload_initiate(
        &mut self,
        index: u16,
        sub_index: u8,
        data: &[u8],
    ) -> Result<[u8; 8], u32> {
        self.index = index;
        self.sub_index = sub_index;
        self.data.clear();
        for &b in data {
            if self.data.push(b).is_err() {
                return Err(ABORT_DATA_TRANSFER);
            }
        }
        self.total_len = data.len();
        self.toggle = false;
        self.segment_count = 0;
        self.direction = TransferDir::Upload;
        self.state = SdoSegState::Initiated;

        // Build upload initiate response: cs=0x41 (size indicated), index, sub, total_len
        let idx = index.to_le_bytes();
        let sz = (data.len() as u32).to_le_bytes();
        Ok([0x41, idx[0], idx[1], sub_index, sz[0], sz[1], sz[2], sz[3]])
    }

    /// Process an upload segment request from client and return the segment response.
    pub fn upload_segment(&mut self, frame: &[u8; 8]) -> Result<[u8; 8], u32> {
        if self.state != SdoSegState::Initiated && self.state != SdoSegState::InProgress {
            return Err(ABORT_GENERAL);
        }
        let cs = frame[0];
        let req_toggle = (cs & 0x10) != 0;
        if self.segment_count > 0 && req_toggle == self.toggle {
            self.state = SdoSegState::Aborted;
            self.abort_code = ABORT_TOGGLE_BIT;
            return Err(ABORT_TOGGLE_BIT);
        }
        self.toggle = req_toggle;

        // Calculate offset into data
        let offset = self.segment_count as usize * 7;
        let remaining = if offset < self.data.len() {
            self.data.len() - offset
        } else {
            0
        };
        let chunk_len = remaining.min(7);
        let last = remaining <= 7;
        let n = 7usize.saturating_sub(chunk_len);

        let mut resp = [0u8; 8];
        // cs = 0x00 | (toggle<<4) | (n<<1) | last
        resp[0] = (if req_toggle { 0x10u8 } else { 0x00u8 })
            | ((n as u8) << 1)
            | (if last { 0x01 } else { 0x00 });

        for i in 0..chunk_len {
            resp[1 + i] = self.data[offset + i];
        }

        self.segment_count += 1;
        self.state = if last {
            SdoSegState::Complete
        } else {
            SdoSegState::InProgress
        };
        Ok(resp)
    }

    // -----------------------------------------------------------------------
    // Abort
    // -----------------------------------------------------------------------

    /// Build an abort frame for a given abort code.
    pub fn build_abort(&self, abort_code: u32) -> [u8; 8] {
        let idx = self.index.to_le_bytes();
        let ac = abort_code.to_le_bytes();
        [
            0x80,
            idx[0],
            idx[1],
            self.sub_index,
            ac[0],
            ac[1],
            ac[2],
            ac[3],
        ]
    }

    /// Process an abort frame from the remote.
    pub fn on_abort(&mut self, frame: &[u8; 8]) {
        if frame[0] == 0x80 {
            self.abort_code = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
            self.state = SdoSegState::Aborted;
        }
    }

    /// Reset to idle state.
    pub fn reset(&mut self) {
        self.state = SdoSegState::Idle;
        self.data.clear();
        self.toggle = false;
        self.segment_count = 0;
        self.abort_code = 0;
    }

    /// Total expected length from initiate frame (0 if not indicated).
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Object index being transferred.
    pub fn index(&self) -> u16 {
        self.index
    }

    /// Object sub-index being transferred.
    pub fn sub_index(&self) -> u8 {
        self.sub_index
    }
}

impl<const BUF: usize> Default for SdoSegmentTransfer<BUF> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_download_init(index: u16, sub: u8, size: u32) -> [u8; 8] {
        let idx = index.to_le_bytes();
        let sz = size.to_le_bytes();
        // cs = 0x21 (download initiate, size indicated, not expedited)
        [0x21, idx[0], idx[1], sub, sz[0], sz[1], sz[2], sz[3]]
    }

    fn make_download_seg(toggle: bool, data: &[u8], last: bool) -> [u8; 8] {
        let mut frame = [0u8; 8];
        let n = 7usize.saturating_sub(data.len());
        frame[0] = (if toggle { 0x10 } else { 0x00 })
            | ((n as u8) << 1)
            | (if last { 0x01 } else { 0x00 });
        for (i, &b) in data.iter().enumerate() {
            frame[1 + i] = b;
        }
        frame
    }

    #[test]
    fn test_download_single_segment() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        let init = make_download_init(0x2001, 0, 5);
        let resp = sdo.download_initiate(&init).unwrap();
        assert_eq!(resp[0], 0x60);
        assert_eq!(sdo.state(), SdoSegState::Initiated);

        let seg = make_download_seg(false, &[1, 2, 3, 4, 5], true);
        let sresp = sdo.download_segment(&seg).unwrap();
        assert_eq!(sresp[0] & 0xE0, 0x20);
        assert_eq!(sdo.state(), SdoSegState::Complete);
        assert_eq!(sdo.data(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_download_multi_segment_toggle() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        let init = make_download_init(0x2002, 0, 14);
        sdo.download_initiate(&init).unwrap();

        // Segment 1 (toggle=false, not last, 7 bytes)
        let s1 = make_download_seg(false, &[10, 11, 12, 13, 14, 15, 16], false);
        sdo.download_segment(&s1).unwrap();
        assert_eq!(sdo.state(), SdoSegState::InProgress);

        // Segment 2 (toggle=true, last, 7 bytes)
        let s2 = make_download_seg(true, &[20, 21, 22, 23, 24, 25, 26], true);
        sdo.download_segment(&s2).unwrap();
        assert_eq!(sdo.state(), SdoSegState::Complete);
        assert_eq!(sdo.data().len(), 14);
    }

    #[test]
    fn test_toggle_bit_error() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        let init = make_download_init(0x2003, 0, 14);
        sdo.download_initiate(&init).unwrap();

        // Send 2 segments with same toggle bit (should fail on 2nd)
        let s1 = make_download_seg(false, &[1, 2, 3, 4, 5, 6, 7], false);
        sdo.download_segment(&s1).unwrap();

        // Send another with same toggle (false) — should error
        let s2 = make_download_seg(false, &[8, 9, 10, 11, 12, 13, 14], true);
        let result = sdo.download_segment(&s2);
        assert_eq!(result, Err(ABORT_TOGGLE_BIT));
        assert_eq!(sdo.state(), SdoSegState::Aborted);
    }

    #[test]
    fn test_upload_transfer() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let resp = sdo.upload_initiate(0x2010, 0, &payload).unwrap();
        assert_eq!(resp[0], 0x41);
        assert_eq!(sdo.state(), SdoSegState::Initiated);

        // Client requests segment 1
        let req1 = [0x60u8, 0, 0, 0, 0, 0, 0, 0]; // toggle=false
        let seg1 = sdo.upload_segment(&req1).unwrap();
        assert_eq!(seg1[1..8], payload[..7]);
        assert_eq!(sdo.state(), SdoSegState::InProgress);

        // Client requests segment 2 with toggle=true
        let req2 = [0x70u8, 0, 0, 0, 0, 0, 0, 0]; // toggle=true
        let seg2 = sdo.upload_segment(&req2).unwrap();
        // last=true, 3 remaining bytes: payload[7]=8, payload[8]=9, payload[9]=10
        assert_eq!(seg2[1], 8); // payload[7]
        assert_eq!(seg2[2], 9); // payload[8]
        assert_eq!(seg2[3], 10); // payload[9]
        assert_eq!(sdo.state(), SdoSegState::Complete);
    }

    #[test]
    fn test_abort_handling() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        sdo.upload_initiate(0x2000, 0, &[1, 2, 3]).unwrap();
        let abort_frame = [0x80u8, 0x00, 0x20, 0x00, 0x00, 0x00, 0x02, 0x06];
        sdo.on_abort(&abort_frame);
        assert_eq!(sdo.state(), SdoSegState::Aborted);
        assert_eq!(sdo.abort_code(), 0x0602_0000);
    }

    #[test]
    fn test_reset() {
        let mut sdo = SdoSegmentTransfer::<64>::new();
        let init = make_download_init(0x2001, 0, 3);
        sdo.download_initiate(&init).unwrap();
        sdo.reset();
        assert_eq!(sdo.state(), SdoSegState::Idle);
        assert_eq!(sdo.data().len(), 0);
        assert_eq!(sdo.segment_count(), 0);
    }
}
