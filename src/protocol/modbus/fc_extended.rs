//! Modbus extended function codes (FC22, FC23, FC24, FC43/14).
//!
//! Implements frame building for less common but standard Modbus functions:
//! - FC22: Mask Write Register
//! - FC23: Read/Write Multiple Registers
//! - FC24: Read FIFO Queue
//! - FC43/14: Read Device Identification

use heapless::Vec;

/// FC22: Mask Write Register function code.
pub const FC_MASK_WRITE: u8 = 0x16;
/// FC23: Read/Write Multiple Registers function code.
pub const FC_READ_WRITE_MULTIPLE: u8 = 0x17;
/// FC24: Read FIFO Queue function code.
pub const FC_READ_FIFO: u8 = 0x18;
/// FC43: Encapsulated Interface Transport.
pub const FC_EIT: u8 = 0x2B;
/// MEI Type 14: Read Device Identification.
pub const MEI_READ_DEVICE_ID: u8 = 0x0E;

/// Device Identification read code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DevIdReadCode {
    /// Basic streaming (objects 0x00–0x02).
    BasicStreamingAccess = 0x01,
    /// Regular streaming (objects 0x00–0x06).
    RegularStreamingAccess = 0x02,
    /// Extended streaming (objects 0x00–0xFF).
    ExtendedStreamingAccess = 0x03,
    /// Individual access to any object.
    IndividualAccess = 0x04,
}

/// Modbus extended frame error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FcExtError {
    /// Invalid address.
    InvalidAddress,
    /// Invalid register count.
    InvalidCount,
    /// Buffer too small.
    BufferOverflow,
    /// Invalid mask value.
    InvalidMask,
}

/// FC22 Mask Write Register request.
///
/// Result = (current_value AND and_mask) OR (or_mask AND NOT and_mask)
#[derive(Debug, Clone, Copy)]
pub struct MaskWriteRequest {
    pub device_id: u8,
    pub address: u16,
    pub and_mask: u16,
    pub or_mask: u16,
}

impl MaskWriteRequest {
    /// Create a masked write request.
    pub fn new(device_id: u8, address: u16, and_mask: u16, or_mask: u16) -> Self {
        Self {
            device_id,
            address,
            and_mask,
            or_mask,
        }
    }

    /// Build the 8-byte RTU request frame (excluding CRC).
    pub fn build_frame(&self) -> [u8; 8] {
        let addr = self.address.to_be_bytes();
        let and = self.and_mask.to_be_bytes();
        let or = self.or_mask.to_be_bytes();
        [
            self.device_id,
            FC_MASK_WRITE,
            addr[0],
            addr[1],
            and[0],
            and[1],
            or[0],
            or[1],
        ]
    }

    /// Apply the mask to a current register value (simulated operation).
    pub fn apply(&self, current: u16) -> u16 {
        (current & self.and_mask) | (self.or_mask & !self.and_mask)
    }

    /// Parse from an 8-byte frame.
    pub fn from_bytes(b: &[u8; 8]) -> Option<Self> {
        if b[1] != FC_MASK_WRITE {
            return None;
        }
        Some(Self {
            device_id: b[0],
            address: u16::from_be_bytes([b[2], b[3]]),
            and_mask: u16::from_be_bytes([b[4], b[5]]),
            or_mask: u16::from_be_bytes([b[6], b[7]]),
        })
    }
}

/// FC23 Read/Write Multiple Registers request.
#[derive(Debug, Clone)]
pub struct ReadWriteMultipleRequest {
    pub device_id: u8,
    pub read_address: u16,
    pub read_count: u16,
    pub write_address: u16,
    pub write_values: Vec<u16, 64>,
}

impl ReadWriteMultipleRequest {
    /// Create a read/write multiple request.
    pub fn new(
        device_id: u8,
        read_address: u16,
        read_count: u16,
        write_address: u16,
    ) -> Result<Self, FcExtError> {
        if read_count == 0 || read_count > 125 {
            return Err(FcExtError::InvalidCount);
        }
        Ok(Self {
            device_id,
            read_address,
            read_count,
            write_address,
            write_values: Vec::new(),
        })
    }

    /// Add a register value to write.
    pub fn add_write_value(&mut self, value: u16) -> Result<(), FcExtError> {
        self.write_values
            .push(value)
            .map_err(|_| FcExtError::BufferOverflow)
    }

    /// Build the RTU request frame into a heapless Vec.
    /// Frame: device_id, FC23, read_addr(2), read_cnt(2),
    ///        write_addr(2), write_cnt(2), byte_cnt(1), write_data(2*n)
    pub fn build_frame<const N: usize>(&self) -> Result<Vec<u8, N>, FcExtError> {
        let mut v: Vec<u8, N> = Vec::new();
        let write_count = self.write_values.len() as u16;
        let byte_count = (write_count * 2) as u8;

        let push = |v: &mut Vec<u8, N>, b: u8| v.push(b).map_err(|_| FcExtError::BufferOverflow);

        push(&mut v, self.device_id)?;
        push(&mut v, FC_READ_WRITE_MULTIPLE)?;
        let ra = self.read_address.to_be_bytes();
        push(&mut v, ra[0])?;
        push(&mut v, ra[1])?;
        let rc = self.read_count.to_be_bytes();
        push(&mut v, rc[0])?;
        push(&mut v, rc[1])?;
        let wa = self.write_address.to_be_bytes();
        push(&mut v, wa[0])?;
        push(&mut v, wa[1])?;
        let wc = write_count.to_be_bytes();
        push(&mut v, wc[0])?;
        push(&mut v, wc[1])?;
        push(&mut v, byte_count)?;
        for &val in &self.write_values {
            let vb = val.to_be_bytes();
            push(&mut v, vb[0])?;
            push(&mut v, vb[1])?;
        }
        Ok(v)
    }
}

/// FC24 Read FIFO Queue request.
#[derive(Debug, Clone, Copy)]
pub struct ReadFifoRequest {
    pub device_id: u8,
    /// Pointer register address.
    pub fifo_pointer: u16,
}

impl ReadFifoRequest {
    /// Create a read FIFO request.
    pub fn new(device_id: u8, fifo_pointer: u16) -> Self {
        Self {
            device_id,
            fifo_pointer,
        }
    }

    /// Build the 4-byte RTU request frame (excluding CRC).
    pub fn build_frame(&self) -> [u8; 4] {
        let ptr = self.fifo_pointer.to_be_bytes();
        [self.device_id, FC_READ_FIFO, ptr[0], ptr[1]]
    }
}

/// FC24 Read FIFO Queue response parser.
#[derive(Debug, Clone)]
pub struct ReadFifoResponse {
    /// FIFO values (up to 31).
    pub values: Vec<u16, 32>,
}

impl ReadFifoResponse {
    /// Parse from raw RTU response bytes (starting after function code).
    /// Expected: byte_count(2), fifo_count(2), values(2*n)
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let fifo_count = u16::from_be_bytes([data[2], data[3]]) as usize;
        if fifo_count > 31 || data.len() < 4 + fifo_count * 2 {
            return None;
        }
        let mut values = Vec::new();
        for i in 0..fifo_count {
            let offset = 4 + i * 2;
            let v = u16::from_be_bytes([data[offset], data[offset + 1]]);
            values.push(v).ok()?;
        }
        Some(Self { values })
    }
}

/// FC43/14 Read Device Identification request.
#[derive(Debug, Clone, Copy)]
pub struct ReadDeviceIdRequest {
    pub device_id: u8,
    pub read_code: DevIdReadCode,
    /// Object ID to start from (0x00 for streaming).
    pub object_id: u8,
}

impl ReadDeviceIdRequest {
    /// Create a read device identification request.
    pub fn new(device_id: u8, read_code: DevIdReadCode, object_id: u8) -> Self {
        Self {
            device_id,
            read_code,
            object_id,
        }
    }

    /// Build the 5-byte RTU request frame (excluding CRC).
    pub fn build_frame(&self) -> [u8; 5] {
        [
            self.device_id,
            FC_EIT,
            MEI_READ_DEVICE_ID,
            self.read_code as u8,
            self.object_id,
        ]
    }
}

/// Parsed device identification object.
#[derive(Debug, Clone, Copy)]
pub struct DevIdObject {
    pub object_id: u8,
    /// Length of the value string.
    pub length: u8,
    /// Value bytes (up to 16).
    pub value: [u8; 16],
}

impl DevIdObject {
    /// Create a new device ID object.
    pub fn new(object_id: u8, value: &[u8]) -> Self {
        let len = value.len().min(16);
        let mut buf = [0u8; 16];
        buf[..len].copy_from_slice(&value[..len]);
        Self {
            object_id,
            length: len as u8,
            value: buf,
        }
    }

    /// Value as byte slice.
    pub fn value_bytes(&self) -> &[u8] {
        &self.value[..self.length as usize]
    }
}

/// FC43/14 Read Device Identification response parser.
#[derive(Debug, Clone)]
pub struct ReadDeviceIdResponse {
    pub mei_type: u8,
    pub read_code: u8,
    pub conformity_level: u8,
    pub more_follows: bool,
    pub next_object_id: u8,
    pub object_count: u8,
    /// Parsed objects (up to 8).
    pub objects: Vec<DevIdObject, 8>,
}

impl ReadDeviceIdResponse {
    /// Parse from raw response bytes (starting after function code byte).
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 6 {
            return None;
        }
        let mei_type = data[0];
        let read_code = data[1];
        let conformity_level = data[2];
        let more_follows = data[3] == 0xFF;
        let next_object_id = data[4];
        let object_count = data[5];

        let mut objects = Vec::new();
        let mut offset = 6usize;
        for _ in 0..object_count {
            if offset + 2 > data.len() {
                return None;
            }
            let oid = data[offset];
            let len = data[offset + 1] as usize;
            offset += 2;
            if offset + len > data.len() {
                return None;
            }
            let obj = DevIdObject::new(oid, &data[offset..offset + len]);
            objects.push(obj).ok()?;
            offset += len;
        }

        Some(Self {
            mei_type,
            read_code,
            conformity_level,
            more_follows,
            next_object_id,
            object_count,
            objects,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_write_build_and_apply() {
        let req = MaskWriteRequest::new(1, 0x0104, 0xFF00, 0x0025);
        let frame = req.build_frame();
        assert_eq!(frame[0], 1);
        assert_eq!(frame[1], FC_MASK_WRITE);
        assert_eq!(u16::from_be_bytes([frame[2], frame[3]]), 0x0104);
        assert_eq!(u16::from_be_bytes([frame[4], frame[5]]), 0xFF00);
        assert_eq!(u16::from_be_bytes([frame[6], frame[7]]), 0x0025);

        // Apply: current=0x12FF, and=0xFF00, or=0x0025
        // result = (0x12FF & 0xFF00) | (0x0025 & 0x00FF) = 0x1200 | 0x0025 = 0x1225
        let result = req.apply(0x12FF);
        assert_eq!(result, 0x1225);
    }

    #[test]
    fn test_mask_write_parse() {
        let req = MaskWriteRequest::new(1, 0x0104, 0xFF00, 0x0025);
        let frame = req.build_frame();
        let parsed = MaskWriteRequest::from_bytes(&frame).unwrap();
        assert_eq!(parsed.address, 0x0104);
        assert_eq!(parsed.and_mask, 0xFF00);
        assert_eq!(parsed.or_mask, 0x0025);
    }

    #[test]
    fn test_read_write_multiple_build() {
        let mut req = ReadWriteMultipleRequest::new(1, 0x0003, 6, 0x000E).unwrap();
        req.add_write_value(0x00FF).unwrap();
        req.add_write_value(0x0BCD).unwrap();
        req.add_write_value(0x1234).unwrap();

        let frame: Vec<u8, 64> = req.build_frame().unwrap();
        assert_eq!(frame[0], 1);
        assert_eq!(frame[1], FC_READ_WRITE_MULTIPLE);
        // read_address = 0x0003
        assert_eq!(u16::from_be_bytes([frame[2], frame[3]]), 0x0003);
        // read_count = 6
        assert_eq!(u16::from_be_bytes([frame[4], frame[5]]), 6);
        // write_address = 0x000E
        assert_eq!(u16::from_be_bytes([frame[6], frame[7]]), 0x000E);
        // write_count = 3
        assert_eq!(u16::from_be_bytes([frame[8], frame[9]]), 3);
        // byte_count = 6
        assert_eq!(frame[10], 6);
    }

    #[test]
    fn test_read_fifo_request() {
        let req = ReadFifoRequest::new(1, 0x04B0);
        let frame = req.build_frame();
        assert_eq!(frame[1], FC_READ_FIFO);
        assert_eq!(u16::from_be_bytes([frame[2], frame[3]]), 0x04B0);
    }

    #[test]
    fn test_read_fifo_response_parse() {
        // byte_count=6, fifo_count=2, val1=0x1234, val2=0x5678
        let data = [0x00u8, 0x06, 0x00, 0x02, 0x12, 0x34, 0x56, 0x78];
        let resp = ReadFifoResponse::parse(&data).unwrap();
        assert_eq!(resp.values.len(), 2);
        assert_eq!(resp.values[0], 0x1234);
        assert_eq!(resp.values[1], 0x5678);
    }

    #[test]
    fn test_read_device_id_request() {
        let req = ReadDeviceIdRequest::new(1, DevIdReadCode::BasicStreamingAccess, 0x00);
        let frame = req.build_frame();
        assert_eq!(frame[1], FC_EIT);
        assert_eq!(frame[2], MEI_READ_DEVICE_ID);
        assert_eq!(frame[3], 0x01); // BasicStreamingAccess
        assert_eq!(frame[4], 0x00);
    }

    #[test]
    fn test_read_device_id_response_parse() {
        // mei=0x0E, read_code=0x01, conformity=0x01, more=0x00, next=0x00, count=2
        // obj[0]: id=0x00, len=8, "Vendor01"
        // obj[1]: id=0x01, len=8, "Product1"
        let v = b"Vendor01";
        let p = b"Product1";
        let mut data = heapless::Vec::<u8, 64>::new();
        let _ = data.push(0x0E);
        let _ = data.push(0x01);
        let _ = data.push(0x01);
        let _ = data.push(0x00); // no more
        let _ = data.push(0x00); // next obj id
        let _ = data.push(0x02); // object count
        let _ = data.push(0x00); // obj id
        let _ = data.push(8); // obj len
        for &b in v {
            let _ = data.push(b);
        }
        let _ = data.push(0x01); // obj id
        let _ = data.push(8); // obj len
        for &b in p {
            let _ = data.push(b);
        }

        let resp = ReadDeviceIdResponse::parse(&data).unwrap();
        assert_eq!(resp.object_count, 2);
        assert_eq!(resp.objects[0].object_id, 0x00);
        assert_eq!(resp.objects[0].value_bytes(), b"Vendor01");
        assert_eq!(resp.objects[1].object_id, 0x01);
    }
}
