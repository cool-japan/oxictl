pub mod ds402;
pub mod emergency;
pub mod lss;
pub mod nmt;
pub mod node;
pub mod object_dict;
pub mod pdo;
pub mod sdo;
pub mod sdo_segment;
pub mod sync;

// ─── Core error / identity types ─────────────────────────────────────────────

/// Top-level error type for the CANopen protocol stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanOpenError {
    /// NMT-layer error.
    Nmt(nmt::NmtError),
    /// Object dictionary error.
    Od(object_dict::OdError),
    /// SDO-layer error.
    Sdo(sdo::SdoError),
    /// PDO-layer error.
    Pdo(pdo::PdoError),
    /// Node ID is outside the valid range 1–127.
    InvalidNodeId,
    /// CAN Object Identifier is outside the valid range for 11-bit IDs.
    InvalidCobId,
}

impl From<nmt::NmtError> for CanOpenError {
    fn from(e: nmt::NmtError) -> Self {
        Self::Nmt(e)
    }
}

impl From<object_dict::OdError> for CanOpenError {
    fn from(e: object_dict::OdError) -> Self {
        Self::Od(e)
    }
}

impl From<sdo::SdoError> for CanOpenError {
    fn from(e: sdo::SdoError) -> Self {
        Self::Sdo(e)
    }
}

impl From<pdo::PdoError> for CanOpenError {
    fn from(e: pdo::PdoError) -> Self {
        Self::Pdo(e)
    }
}

/// CANopen node identifier newtype.
///
/// Valid range is 1–127 inclusive.  Node ID 0 is the NMT broadcast address
/// and is not a valid unicast node ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u8);

impl NodeId {
    /// Minimum valid node ID.
    pub const MIN: u8 = 1;
    /// Maximum valid node ID.
    pub const MAX: u8 = 127;

    /// Construct a `NodeId`, returning `None` if `id` is outside 1–127.
    pub const fn new(id: u8) -> Option<Self> {
        if id >= Self::MIN && id <= Self::MAX {
            Some(Self(id))
        } else {
            None
        }
    }

    /// Construct a `NodeId` without range checking.
    ///
    /// # Safety
    /// The caller must ensure `id` is in the range 1–127.
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    /// Return the raw node ID byte.
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Default TPDO1 COB-ID for this node (0x180 + node_id).
    pub fn tpdo1_cob_id(self) -> CobId {
        CobId(0x180 + self.0 as u32)
    }

    /// Default RPDO1 COB-ID for this node (0x200 + node_id).
    pub fn rpdo1_cob_id(self) -> CobId {
        CobId(0x200 + self.0 as u32)
    }

    /// SDO server receive COB-ID (0x600 + node_id).
    pub fn sdo_rx_cob_id(self) -> CobId {
        CobId(0x600 + self.0 as u32)
    }

    /// SDO server transmit COB-ID (0x580 + node_id).
    pub fn sdo_tx_cob_id(self) -> CobId {
        CobId(0x580 + self.0 as u32)
    }

    /// NMT heartbeat COB-ID (0x700 + node_id).
    pub fn heartbeat_cob_id(self) -> CobId {
        CobId(0x700 + self.0 as u32)
    }

    /// Emergency COB-ID (0x080 + node_id).
    pub fn emcy_cob_id(self) -> CobId {
        CobId(0x080 + self.0 as u32)
    }
}

/// CAN Object Identifier (COB-ID) newtype.
///
/// For standard 11-bit CAN identifiers the valid range is 0x000–0x7FF.
/// The CANopen specification also allows 29-bit extended frame IDs; this
/// type stores the full 32-bit value but validation is 11-bit by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CobId(pub u32);

impl CobId {
    /// NMT command frame COB-ID (always 0x000).
    pub const NMT: Self = Self(0x000);
    /// SYNC COB-ID (default 0x080).
    pub const SYNC: Self = Self(0x080);
    /// Emergency base COB-ID — add node_id for unicast (0x081–0x0FF).
    pub const EMCY_BASE: Self = Self(0x080);
    /// Time-stamp COB-ID (0x100).
    pub const TIMESTAMP: Self = Self(0x100);
    /// SDO server transmit base (0x580).
    pub const SDO_TX_BASE: Self = Self(0x580);
    /// SDO server receive base (0x600).
    pub const SDO_RX_BASE: Self = Self(0x600);
    /// NMT heartbeat base (0x700).
    pub const HEARTBEAT_BASE: Self = Self(0x700);

    /// Construct a `CobId`, returning `Err` for values above the 11-bit limit.
    pub fn new_standard(id: u32) -> Result<Self, CanOpenError> {
        if id > 0x7FF {
            Err(CanOpenError::InvalidCobId)
        } else {
            Ok(Self(id))
        }
    }

    /// Construct a `CobId` from any 32-bit value (extended frame or vendor).
    pub const fn new_extended(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw CAN identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use ds402::{DriveState, Ds402StateMachine};
pub use emergency::{EmergencyEvent, EmergencyProducer};
pub use lss::{LssClientCo, LssClientCoState};
pub use nmt::{
    HeartbeatFrame, HeartbeatProducer, NmtCommand, NmtController, NmtError, NmtMessage, NmtState,
    NmtStateMachine,
};
pub use node::CanOpenNode;
pub use object_dict::{
    AccessType, DataType, ObjectDict, OdEntry, OdEntryValue, OdError, OdIndex, OdValue, StaticOd,
};
pub use pdo::{
    CanFrame, Pdo, PdoComm, PdoError, PdoMapEntry, PdoMapping, RpdoConfig, RpdoConsumer,
    TpdoConfig, TpdoProducer,
};
pub use sdo::{SdoAbortCode, SdoError, SdoFrame, SdoServer};
pub use sdo_segment::SdoSegmentTransfer;
pub use sync::{SyncConsumer, SyncProducer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_valid_range() {
        assert!(NodeId::new(1).is_some());
        assert!(NodeId::new(127).is_some());
        assert!(NodeId::new(0).is_none());
        assert!(NodeId::new(128).is_none());
    }

    #[test]
    fn node_id_cob_ids() {
        let node = NodeId::new_unchecked(5);
        assert_eq!(node.tpdo1_cob_id().get(), 0x185);
        assert_eq!(node.rpdo1_cob_id().get(), 0x205);
        assert_eq!(node.sdo_rx_cob_id().get(), 0x605);
        assert_eq!(node.sdo_tx_cob_id().get(), 0x585);
        assert_eq!(node.heartbeat_cob_id().get(), 0x705);
        assert_eq!(node.emcy_cob_id().get(), 0x085);
    }

    #[test]
    fn cob_id_standard_range() {
        assert!(CobId::new_standard(0x7FF).is_ok());
        assert!(CobId::new_standard(0x800).is_err());
    }

    #[test]
    fn cob_id_constants() {
        assert_eq!(CobId::NMT.get(), 0x000);
        assert_eq!(CobId::SYNC.get(), 0x080);
        assert_eq!(CobId::SDO_TX_BASE.get(), 0x580);
        assert_eq!(CobId::SDO_RX_BASE.get(), 0x600);
        assert_eq!(CobId::HEARTBEAT_BASE.get(), 0x700);
    }

    #[test]
    fn canopen_error_from_nmt_error() {
        let e = CanOpenError::from(nmt::NmtError::UnknownCommand(0xFF));
        matches!(e, CanOpenError::Nmt(_));
    }

    #[test]
    fn canopen_error_from_od_error() {
        let e = CanOpenError::from(object_dict::OdError::NotFound);
        matches!(e, CanOpenError::Od(_));
    }
}
