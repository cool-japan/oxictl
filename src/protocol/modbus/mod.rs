pub mod diagnostics;
pub mod fc_extended;
pub mod pdu;
pub mod register;
pub mod register_map;
pub mod rtu;
pub mod server;
pub mod tcp;

// ─── Common types ─────────────────────────────────────────────────────────────

/// Modbus device (slave) address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceAddr(pub u8);

impl DeviceAddr {
    /// Broadcast address.
    pub const BROADCAST: Self = Self(0x00);

    /// Maximum unicast address.
    pub const MAX: Self = Self(0xF7);

    /// Construct from a raw byte.  Panics in debug if addr is 0xFF (reserved).
    pub const fn new(addr: u8) -> Self {
        Self(addr)
    }

    /// Return the raw address byte.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Modbus register address (0-based, 0x0000–0xFFFF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisterAddr(pub u16);

impl RegisterAddr {
    /// Construct from a raw u16.
    pub const fn new(addr: u16) -> Self {
        Self(addr)
    }

    /// Return the raw address.
    pub const fn get(self) -> u16 {
        self.0
    }
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use diagnostics::{DiagnosticCounters, DiagnosticServer};
pub use fc_extended::{
    MaskWriteRequest, ReadDeviceIdRequest, ReadFifoRequest, ReadWriteMultipleRequest,
};
pub use pdu::{decode_response, encode_request, ExceptionCode, Request, Response};
pub use register::{ModbusError, RegisterMap};
pub use register_map::{CoilBank, RegisterBank};
pub use rtu::{
    crc16, decode_rtu, encode_rtu, silent_interval_us, FunctionCode, RtuFrame, RtuMaster,
    RtuMasterState, RtuWriter,
};
pub use server::{ModbusPdu, ModbusServer};
pub use tcp::{decode_tcp, encode_tcp, MbapHeader, TcpFrame, TcpSession};
