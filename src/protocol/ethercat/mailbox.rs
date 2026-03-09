//! EtherCAT Mailbox protocol - acyclic communication channel.
//!
//! The mailbox provides reliable, acyclic communication for SDO, FoE,
//! EoE, and SoE protocols. Each mailbox channel buffers frames in a
//! heapless Deque for no_alloc operation.

use heapless::Deque;

/// Mailbox protocol type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MailboxType {
    /// EtherNet over EtherCAT.
    EoE = 2,
    /// CANopen over EtherCAT.
    CoE = 3,
    /// File access over EtherCAT.
    FoE = 4,
    /// Servo Drive Profile over EtherCAT.
    SoE = 10,
}

impl MailboxType {
    /// Parse from raw byte value.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            2 => Some(Self::EoE),
            3 => Some(Self::CoE),
            4 => Some(Self::FoE),
            10 => Some(Self::SoE),
            _ => None,
        }
    }
}

/// EtherCAT mailbox header (6 bytes).
#[derive(Debug, Clone, Copy)]
pub struct MailboxHeader {
    /// Data length (excluding header).
    pub length: u16,
    /// Slave address.
    pub address: u16,
    /// Channel priority (0–3).
    pub priority: u8,
    /// Mailbox type.
    pub mbx_type: MailboxType,
    /// Counter (1–7, wraps; 0 = not used).
    pub counter: u8,
}

impl MailboxHeader {
    /// Create a new mailbox header.
    pub fn new(
        length: u16,
        address: u16,
        priority: u8,
        mbx_type: MailboxType,
        counter: u8,
    ) -> Self {
        Self {
            length,
            address,
            priority,
            mbx_type,
            counter,
        }
    }

    /// Serialize header to 6 bytes.
    pub fn to_bytes(&self) -> [u8; 6] {
        let len = self.length.to_le_bytes();
        let addr = self.address.to_le_bytes();
        // Byte 4: channel(bit 7:6)=0, priority(bit 5:4), type(bit 3:0)
        let ctrl = ((self.priority & 0x03) << 4) | (self.mbx_type as u8 & 0x0F);
        // Byte 5: counter(bit 6:4), reserved(bit 3:0)
        let cnt = (self.counter & 0x07) << 4;
        [len[0], len[1], addr[0], addr[1], ctrl, cnt]
    }

    /// Parse header from 6 bytes.
    pub fn from_bytes(b: &[u8; 6]) -> Option<Self> {
        let length = u16::from_le_bytes([b[0], b[1]]);
        let address = u16::from_le_bytes([b[2], b[3]]);
        let priority = (b[4] >> 4) & 0x03;
        let mbx_type = MailboxType::from_u8(b[4] & 0x0F)?;
        let counter = (b[5] >> 4) & 0x07;
        Some(Self {
            length,
            address,
            priority,
            mbx_type,
            counter,
        })
    }
}

/// A fixed-size mailbox frame (header + up to 128 bytes of data).
#[derive(Debug, Clone, Copy)]
pub struct MailboxFrame {
    pub header: MailboxHeader,
    pub data: [u8; 128],
    pub data_len: usize,
}

impl MailboxFrame {
    /// Create a new mailbox frame.
    pub fn new(header: MailboxHeader, data: &[u8]) -> Self {
        let mut buf = [0u8; 128];
        let len = data.len().min(128);
        buf[..len].copy_from_slice(&data[..len]);
        Self {
            header,
            data: buf,
            data_len: len,
        }
    }

    /// Serialize the full frame (header + data) into a buffer.
    /// Returns number of bytes written.
    pub fn serialize(&self, out: &mut [u8; 134]) -> usize {
        let hdr = self.header.to_bytes();
        out[..6].copy_from_slice(&hdr);
        out[6..6 + self.data_len].copy_from_slice(&self.data[..self.data_len]);
        6 + self.data_len
    }
}

/// Errors for mailbox operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailboxError {
    BufferFull,
    BufferEmpty,
    InvalidType,
    InvalidLength,
    ParseError,
}

/// Buffered mailbox channel using heapless Deque.
///
/// `N` is the maximum number of frames that can be queued.
pub struct MailboxChannel<const N: usize> {
    tx_queue: Deque<MailboxFrame, N>,
    rx_queue: Deque<MailboxFrame, N>,
    counter: u8,
    address: u16,
}

impl<const N: usize> MailboxChannel<N> {
    /// Create a new mailbox channel for the given slave address.
    pub fn new(address: u16) -> Self {
        Self {
            tx_queue: Deque::new(),
            rx_queue: Deque::new(),
            counter: 1,
            address,
        }
    }

    /// Enqueue a frame for transmission.
    pub fn send(&mut self, mbx_type: MailboxType, data: &[u8]) -> Result<(), MailboxError> {
        if data.len() > 128 {
            return Err(MailboxError::InvalidLength);
        }
        let hdr = MailboxHeader::new(data.len() as u16, self.address, 0, mbx_type, self.counter);
        self.counter = if self.counter >= 7 {
            1
        } else {
            self.counter + 1
        };
        let frame = MailboxFrame::new(hdr, data);
        self.tx_queue
            .push_back(frame)
            .map_err(|_| MailboxError::BufferFull)
    }

    /// Dequeue a frame from the transmit queue.
    pub fn poll_tx(&mut self) -> Option<MailboxFrame> {
        self.tx_queue.pop_front()
    }

    /// Accept a received frame into the receive queue.
    pub fn receive(&mut self, frame: MailboxFrame) -> Result<(), MailboxError> {
        self.rx_queue
            .push_back(frame)
            .map_err(|_| MailboxError::BufferFull)
    }

    /// Dequeue a received frame.
    pub fn poll_rx(&mut self) -> Option<MailboxFrame> {
        self.rx_queue.pop_front()
    }

    /// Number of frames pending in TX queue.
    pub fn tx_pending(&self) -> usize {
        self.tx_queue.len()
    }

    /// Number of frames pending in RX queue.
    pub fn rx_pending(&self) -> usize {
        self.rx_queue.len()
    }

    /// Slave address.
    pub fn address(&self) -> u16 {
        self.address
    }
}

/// CoE (CANopen over EtherCAT) SDO message wrapper.
#[derive(Debug, Clone, Copy)]
pub struct CoEMessage {
    /// CoE service type (SDO request=0x200, response=0x300, etc.).
    pub service: u16,
    /// CoE command specifier.
    pub command: u8,
    /// Object dictionary index.
    pub index: u16,
    /// Sub-index.
    pub sub_index: u8,
    /// Data payload (up to 4 bytes for expedited).
    pub data: [u8; 4],
    /// Number of valid data bytes.
    pub data_size: u8,
}

impl CoEMessage {
    /// CoE service types.
    pub const SVC_SDO_REQ: u16 = 0x200;
    pub const SVC_SDO_RESP: u16 = 0x300;
    pub const SVC_EMCY: u16 = 0x100;

    /// Create an SDO download request (expedited, ≤4 bytes).
    pub fn sdo_download_req(index: u16, sub_index: u8, data: &[u8]) -> Self {
        let size = data.len().min(4) as u8;
        let mut buf = [0u8; 4];
        buf[..size as usize].copy_from_slice(&data[..size as usize]);
        // cs = 0x23 | ((4-n)<<2) for n=size
        let cs = 0x23u8 | ((4u8.saturating_sub(size)) << 2);
        Self {
            service: Self::SVC_SDO_REQ,
            command: cs,
            index,
            sub_index,
            data: buf,
            data_size: size,
        }
    }

    /// Create an SDO upload request (read).
    pub fn sdo_upload_req(index: u16, sub_index: u8) -> Self {
        Self {
            service: Self::SVC_SDO_REQ,
            command: 0x40,
            index,
            sub_index,
            data: [0u8; 4],
            data_size: 0,
        }
    }

    /// Serialize to bytes for embedding in a mailbox frame.
    /// Returns 10 bytes: [svc_lo, svc_hi, cmd, idx_lo, idx_hi, sub, d0, d1, d2, d3]
    pub fn to_bytes(&self) -> [u8; 10] {
        let svc = self.service.to_le_bytes();
        let idx = self.index.to_le_bytes();
        [
            svc[0],
            svc[1],
            self.command,
            idx[0],
            idx[1],
            self.sub_index,
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
        ]
    }

    /// Parse from 10-byte payload.
    pub fn from_bytes(b: &[u8; 10]) -> Self {
        Self {
            service: u16::from_le_bytes([b[0], b[1]]),
            command: b[2],
            index: u16::from_le_bytes([b[3], b[4]]),
            sub_index: b[5],
            data: [b[6], b[7], b[8], b[9]],
            data_size: 4,
        }
    }

    /// Check if this is an SDO abort.
    pub fn is_abort(&self) -> bool {
        self.command == 0x80
    }

    /// Get abort code from data field.
    pub fn abort_code(&self) -> u32 {
        u32::from_le_bytes(self.data)
    }

    /// Create an SDO abort response.
    pub fn sdo_abort(index: u16, sub_index: u8, abort_code: u32) -> Self {
        Self {
            service: Self::SVC_SDO_RESP,
            command: 0x80,
            index,
            sub_index,
            data: abort_code.to_le_bytes(),
            data_size: 4,
        }
    }
}

/// FoE (File over EtherCAT) operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FoeOpCode {
    Read = 1,
    Write = 2,
    Data = 3,
    Ack = 4,
    Error = 5,
    Busy = 6,
}

/// FoE message header.
#[derive(Debug, Clone, Copy)]
pub struct FoeMessage {
    pub op_code: FoeOpCode,
    pub reserved: u8,
    pub packet_no: u16,
}

impl FoeMessage {
    /// Create a write initiation message.
    pub fn write_init(filename_len: u16) -> Self {
        Self {
            op_code: FoeOpCode::Write,
            reserved: 0,
            packet_no: filename_len,
        }
    }

    /// Create an ack message for a given packet number.
    pub fn ack(packet_no: u16) -> Self {
        Self {
            op_code: FoeOpCode::Ack,
            reserved: 0,
            packet_no,
        }
    }

    /// Serialize to 4 bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        let pn = self.packet_no.to_le_bytes();
        [self.op_code as u8, self.reserved, pn[0], pn[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mailbox_header_roundtrip() {
        let hdr = MailboxHeader::new(10, 0x0001, 0, MailboxType::CoE, 3);
        let bytes = hdr.to_bytes();
        let parsed = MailboxHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.length, 10);
        assert_eq!(parsed.address, 0x0001);
        assert_eq!(parsed.mbx_type, MailboxType::CoE);
        assert_eq!(parsed.counter, 3);
    }

    #[test]
    fn test_mailbox_channel_send_receive() {
        let mut ch = MailboxChannel::<4>::new(0x0001);
        assert!(ch
            .send(
                MailboxType::CoE,
                &[0x23, 0x40, 0x60, 0x00, 0x0F, 0x00, 0x00, 0x00]
            )
            .is_ok());
        assert_eq!(ch.tx_pending(), 1);
        let frame = ch.poll_tx().unwrap();
        assert_eq!(frame.header.mbx_type, MailboxType::CoE);
        assert_eq!(ch.tx_pending(), 0);
    }

    #[test]
    fn test_coe_message_sdo_download() {
        let msg = CoEMessage::sdo_download_req(0x6040, 0, &[0x0F, 0x00]);
        assert_eq!(msg.service, CoEMessage::SVC_SDO_REQ);
        assert_eq!(msg.index, 0x6040);
        assert_eq!(msg.sub_index, 0);
        let bytes = msg.to_bytes();
        let parsed = CoEMessage::from_bytes(&bytes);
        assert_eq!(parsed.index, 0x6040);
    }

    #[test]
    fn test_mailbox_channel_buffer_full() {
        let mut ch = MailboxChannel::<2>::new(0x0001);
        assert!(ch.send(MailboxType::CoE, &[1, 2, 3]).is_ok());
        assert!(ch.send(MailboxType::CoE, &[4, 5, 6]).is_ok());
        assert_eq!(
            ch.send(MailboxType::CoE, &[7, 8, 9]),
            Err(MailboxError::BufferFull)
        );
    }

    #[test]
    fn test_coe_abort() {
        let abort = CoEMessage::sdo_abort(0x6040, 0, 0x0602_0000);
        assert!(abort.is_abort());
        assert_eq!(abort.abort_code(), 0x0602_0000);
    }

    #[test]
    fn test_mailbox_type_from_u8() {
        assert_eq!(MailboxType::from_u8(3), Some(MailboxType::CoE));
        assert_eq!(MailboxType::from_u8(4), Some(MailboxType::FoE));
        assert_eq!(MailboxType::from_u8(99), None);
    }
}
