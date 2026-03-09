//! CANopen node — object dictionary and communication objects.

use super::nmt::{NmtMessage, NmtState, NmtStateMachine};

/// CANopen object dictionary entry.
#[derive(Debug, Clone, Copy)]
pub struct OdEntry {
    pub index: u16,
    pub sub_index: u8,
    pub data: u32,
    pub read_only: bool,
}

/// CANopen node with object dictionary.
pub struct CanOpenNode<const OD_SIZE: usize> {
    pub nmt: NmtStateMachine,
    od: [Option<OdEntry>; OD_SIZE],
    od_count: usize,
}

impl<const OD_SIZE: usize> CanOpenNode<OD_SIZE> {
    pub fn new(node_id: u8) -> Self {
        Self {
            nmt: NmtStateMachine::new(node_id),
            od: core::array::from_fn(|_| None),
            od_count: 0,
        }
    }

    pub fn node_id(&self) -> u8 {
        self.nmt.node_id
    }
    pub fn state(&self) -> NmtState {
        self.nmt.state
    }

    /// Boot up the node (Init → Pre-Op).
    pub fn boot_up(&mut self) {
        self.nmt.boot_up();
    }

    /// Apply NMT command.
    pub fn apply_nmt(&mut self, msg: &NmtMessage) -> bool {
        self.nmt.process(msg)
    }

    /// Add object dictionary entry.
    pub fn add_od(&mut self, index: u16, sub_index: u8, value: u32, read_only: bool) -> bool {
        if self.od_count >= OD_SIZE {
            return false;
        }
        self.od[self.od_count] = Some(OdEntry {
            index,
            sub_index,
            data: value,
            read_only,
        });
        self.od_count += 1;
        true
    }

    fn find_od(&self, index: u16, sub_index: u8) -> Option<usize> {
        self.od[..self.od_count].iter().position(|e| {
            e.map(|e| e.index == index && e.sub_index == sub_index)
                .unwrap_or(false)
        })
    }

    /// Read OD entry (SDO upload simulation).
    pub fn read_od(&self, index: u16, sub_index: u8) -> Option<u32> {
        self.find_od(index, sub_index)
            .and_then(|i| self.od[i].map(|e| e.data))
    }

    /// Write OD entry (SDO download simulation).
    pub fn write_od(&mut self, index: u16, sub_index: u8, value: u32) -> bool {
        match self.find_od(index, sub_index) {
            None => false,
            Some(i) => {
                if let Some(e) = &mut self.od[i] {
                    if e.read_only {
                        return false;
                    }
                    e.data = value;
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::nmt::NmtCommand;
    use super::*;

    #[test]
    fn canopen_node_boot_up() {
        let mut node = CanOpenNode::<16>::new(5);
        node.boot_up();
        assert_eq!(node.state(), NmtState::PreOperational);
    }

    #[test]
    fn canopen_node_od_rw() {
        let mut node = CanOpenNode::<16>::new(1);
        node.add_od(0x6040, 0, 0x0000, false);
        node.write_od(0x6040, 0, 0x000F);
        assert_eq!(node.read_od(0x6040, 0), Some(0x000F));
    }

    #[test]
    fn canopen_node_od_read_only() {
        let mut node = CanOpenNode::<8>::new(1);
        node.add_od(0x1000, 0, 0x00020004, true);
        assert!(!node.write_od(0x1000, 0, 0x0));
        assert_eq!(node.read_od(0x1000, 0), Some(0x00020004));
    }

    #[test]
    fn canopen_node_nmt_command() {
        let mut node = CanOpenNode::<4>::new(3);
        node.boot_up();
        node.apply_nmt(&NmtMessage::broadcast(NmtCommand::StartRemoteNode));
        assert_eq!(node.state(), NmtState::Operational);
    }
}
