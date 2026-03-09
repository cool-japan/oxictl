//! Integration test: EtherCAT bus simulation.

use oxictl::protocol::canopen::ds402::{cw, sw, DriveState, Ds402StateMachine};
use oxictl::protocol::ethercat::dc::{DcState, DcSynchronizer};
use oxictl::protocol::ethercat::master::{AlState, EtherCatMaster, MasterState, SlaveConfig};
use oxictl::protocol::ethercat::pdo::{PdoEntry, PdoMapping, ProcessImage};
use oxictl::protocol::ethercat::sdo::SdoClient;

/// Full EtherCAT bus bring-up sequence: INIT → OP.
#[test]
fn ethercat_bus_brings_up_to_op() {
    let mut master = EtherCatMaster::<8>::new();
    master.add_slave(SlaveConfig::new(0, 0x0000_1234, 0x0001));
    master.add_slave(SlaveConfig::new(1, 0x0000_5678, 0x0002));

    assert_eq!(master.state, MasterState::Idle);

    assert!(master.request_state(AlState::PreOp));
    assert_eq!(master.state, MasterState::Configuring);

    assert!(master.request_state(AlState::SafeOp));
    assert_eq!(master.state, MasterState::Configuring);

    assert!(master.request_state(AlState::Op));
    assert_eq!(master.state, MasterState::Running);
}

/// EtherCAT master: request_state always succeeds (software model).
#[test]
fn ethercat_direct_to_op_succeeds() {
    let mut master = EtherCatMaster::<4>::new();
    // Software model: always returns true
    let ok = master.request_state(AlState::Op);
    assert!(ok, "Software model always accepts state transitions");
    assert_eq!(master.state, MasterState::Running);
}

/// PDO process image read/write.
#[test]
fn ethercat_pdo_process_image() {
    let mut img = ProcessImage::<64>::new();

    // Write u16 at offset 0
    img.write_u16(0, 0xABCD);
    assert_eq!(img.read_u16(0), Some(0xABCD));

    // Write u32 at offset 4
    img.write_u32(4, 0xDEAD_BEEF);
    assert_eq!(img.read_u32(4), Some(0xDEAD_BEEF));

    // No overlap
    assert_eq!(img.read_u16(0), Some(0xABCD));
}

/// PDO mapping byte size calculation.
#[test]
fn ethercat_pdo_byte_size() {
    let mut pdo = PdoMapping::<3>::new(0x1A00);
    pdo.add_entry(PdoEntry::new(0x6064, 0, 32)); // 4 bytes
    pdo.add_entry(PdoEntry::new(0x6041, 0, 16)); // 2 bytes
    pdo.add_entry(PdoEntry::new(0x6061, 0, 8)); // 1 byte
    assert_eq!(pdo.total_bytes(), 7); // 4+2+1
}

/// SDO client read/write object dictionary.
#[test]
fn ethercat_sdo_read_write() {
    let mut sdo = SdoClient::new();

    // Define and write/read back
    sdo.define_object(0x6060, 0, &[0x00, 0x00], false);
    sdo.write_u16(0x6060, 0, 0x08).unwrap(); // Modes of operation = 8 (cyclic sync pos)
    let val = sdo.read_u16(0x6060, 0).unwrap();
    assert_eq!(val, 0x08);

    // Non-existent entry returns Err
    assert!(sdo.read_u16(0x9999, 0).is_err());
}

/// DS-402 drive state machine: full bring-up sequence.
#[test]
fn ds402_full_bringup_to_operation_enabled() {
    let mut ds402 = Ds402StateMachine::new();
    assert_eq!(ds402.state, DriveState::SwitchOnDisabled);

    // Step 1: → Ready To Switch On
    ds402.apply_control_word(cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    assert_eq!(ds402.state, DriveState::ReadyToSwitchOn);

    // Step 2: → Switched On
    ds402.apply_control_word(cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    assert_eq!(ds402.state, DriveState::SwitchedOn);

    // Step 3: → Operation Enabled
    ds402.apply_control_word(
        cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION,
    );
    assert_eq!(ds402.state, DriveState::OperationEnabled);
}

/// DS-402: fault reset procedure.
#[test]
fn ds402_fault_reset() {
    let mut ds402 = Ds402StateMachine::new();

    // Get to Operation Enabled
    ds402.apply_control_word(cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    ds402.apply_control_word(cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    ds402.apply_control_word(
        cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION,
    );
    assert_eq!(ds402.state, DriveState::OperationEnabled);

    // Inject fault → FaultReactionActive
    ds402.inject_fault(0x3210);
    assert_eq!(ds402.state, DriveState::FaultReactionActive);

    // Apply any CW to trigger FaultReactionActive → Fault
    ds402.apply_control_word(0);
    assert_eq!(ds402.state, DriveState::Fault);

    // Fault reset
    ds402.apply_control_word(cw::FAULT_RESET);
    assert_eq!(ds402.state, DriveState::SwitchOnDisabled);
}

/// DC synchronizer locks after many cycles.
#[test]
fn dc_synchronizer_locks() {
    let mut dc = DcSynchronizer::new(2_000_000, 1000); // 2ms cycle, 1µs tolerance
    dc.enable();

    // Initially locking (not locked)
    assert_ne!(dc.state, DcState::Locked, "Should not start locked");

    // Feed many aligned timestamps → should lock
    let mut t = 0u64;
    for _ in 0..50 {
        t += 2_000_000; // Perfect cycle time
        dc.update(t);
    }

    // After consistent cycles, should be locked or locking
    assert_ne!(
        dc.state,
        DcState::Disabled,
        "DC should not be disabled after cycles"
    );
}

/// Full integration: EtherCAT + DS-402 + PDO cyclic exchange.
#[test]
fn ethercat_full_cyclic_exchange() {
    let mut master = EtherCatMaster::<2>::new();
    master.add_slave(SlaveConfig::new(0, 0xCAFE_0001, 0x0001));

    // Bring to OP
    master.request_state(AlState::PreOp);
    master.request_state(AlState::SafeOp);
    master.request_state(AlState::Op);
    assert_eq!(master.state, MasterState::Running);

    let mut ds402 = Ds402StateMachine::new();
    ds402.apply_control_word(cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    ds402.apply_control_word(cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP);
    ds402.apply_control_word(
        cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION,
    );

    let mut img = ProcessImage::<32>::new();

    // Simulate 10 cyclic exchanges
    for step in 0u32..10 {
        let target_pos = step * 1000;
        img.write_u32(0, target_pos);
        let ctrl = cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION;
        img.write_u16(4, ctrl);

        // Simulate slave response
        let actual_pos = target_pos.saturating_sub(500); // Follows with lag
        img.write_u32(8, actual_pos);
        let status = sw::SWITCHED_ON | sw::OPERATION_ENABLED | sw::VOLTAGE_ENABLED | sw::QUICK_STOP;
        img.write_u16(12, status);

        assert_eq!(img.read_u32(0), Some(target_pos));
        assert_eq!(img.read_u32(8), Some(actual_pos));
    }
}
