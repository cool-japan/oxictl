//! EtherCAT Servo Drive Simulation.
//!
//! Demonstrates EtherCAT master state machine + PDO + DS-402 drive profile
//! with a simulated servo axis.

use oxictl::protocol::canopen::ds402::{cw, sw, Ds402StateMachine};
use oxictl::protocol::ethercat::dc::DcSynchronizer;
use oxictl::protocol::ethercat::master::{AlState, EtherCatMaster, MasterState, SlaveConfig};
use oxictl::protocol::ethercat::pdo::{PdoEntry, PdoMapping, ProcessImage};
use oxictl::protocol::ethercat::sdo::SdoClient;

fn main() {
    println!("=== EtherCAT Servo Simulation ===\n");

    // 1. Create EtherCAT master
    let mut master = EtherCatMaster::<4>::new();
    master.add_slave(SlaveConfig::new(0, 0x0000_22D2, 0x0000_0001)); // Servo drive

    println!("[Master] State: {:?}", master.state);

    // 2. Bring up bus: INIT → PREOP → SAFEOP → OP
    let transitions = [AlState::PreOp, AlState::SafeOp, AlState::Op];
    for &target in &transitions {
        if master.request_state(target) {
            println!("[Master] → {:?}", master.state);
        } else {
            println!("[Master] Failed to reach {:?}", target);
            return;
        }
    }

    // 3. Configure PDOs for position control
    // TxPDO: Actual position (0x6064) + Status word (0x6041)
    let mut tx_pdo = PdoMapping::<2>::new(0x1A00);
    tx_pdo.add_entry(PdoEntry::new(0x6064, 0x00, 32)); // Actual position (32-bit)
    tx_pdo.add_entry(PdoEntry::new(0x6041, 0x00, 16)); // Status word (16-bit)

    // RxPDO: Target position (0x607A) + Control word (0x6040)
    let mut rx_pdo = PdoMapping::<2>::new(0x1600);
    rx_pdo.add_entry(PdoEntry::new(0x607A, 0x00, 32)); // Target position (32-bit)
    rx_pdo.add_entry(PdoEntry::new(0x6040, 0x00, 16)); // Control word (16-bit)

    println!("\n[PDO] TxPDO size: {} bytes", tx_pdo.total_bytes());
    println!("[PDO] RxPDO size: {} bytes", rx_pdo.total_bytes());

    // 4. SDO configuration
    let mut sdo = SdoClient::new();
    sdo.define_object(0x6098, 0, &[0x00], false); // Homing method
    sdo.define_object(0x6099, 1, &[0x00, 0x00], false); // Homing speed
    sdo.write_u16(0x6098, 0, 0x0023).ok(); // Homing method = 35
    sdo.write_u16(0x6099, 1, 1000).ok(); // Homing speed
    println!(
        "\n[SDO] Homing method set: 0x{:02X}",
        sdo.read_u16(0x6098, 0).unwrap_or(0)
    );

    // 5. Distributed Clock setup
    let mut dc = DcSynchronizer::new(2_000_000, 500); // 2ms cycle, 500ns tolerance
    dc.enable();
    println!("\n[DC] State: {:?}", dc.state);

    // Simulate DC locking (drift compensation)
    let mut t = 0u64;
    for _ in 0..20 {
        t += 2_000_000 + 800; // Slight drift
        dc.update(t);
    }
    println!("[DC] After locking: {:?}", dc.state);

    // 6. DS-402 state machine: bring drive to Operation Enabled
    let mut ds402 = Ds402StateMachine::new();
    println!("\n[DS402] Initial state: {:?}", ds402.state);

    let drive_sequence = [
        cw::ENABLE_VOLTAGE | cw::QUICK_STOP, // → Ready To Switch On
        cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP, // → Switched On
        cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION, // → Op Enabled
    ];

    for &ctrl_word in &drive_sequence {
        ds402.apply_control_word(ctrl_word);
        println!("[DS402] CW=0x{:04X} → State: {:?}", ctrl_word, ds402.state);
    }

    // 7. Simulate cyclic position control loop
    println!("\n[Loop] Starting cyclic position control...");
    println!("step,target_pos,actual_pos,status_word");

    let mut process_image = ProcessImage::<64>::new();
    let mut actual_pos_inc: i32 = 0;
    let target_positions = [0i32, 100_000, 200_000, 150_000, 0]; // encoder increments

    for (step, &target) in target_positions.iter().enumerate() {
        // Write target position to process image (RxPDO offset 0)
        process_image.write_u32(0, target as u32);
        // Write control word (Op Enabled, no halt)
        let ctrl = cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION;
        process_image.write_u16(4, ctrl);

        // Simulate drive following (simplified: move 20% closer per step)
        actual_pos_inc += (target - actual_pos_inc) / 5;

        // Update process image TxPDO
        process_image.write_u32(6, actual_pos_inc as u32); // Actual position
        let sw_val = sw::SWITCHED_ON | sw::OPERATION_ENABLED | sw::VOLTAGE_ENABLED | sw::QUICK_STOP;
        process_image.write_u16(10, sw_val);

        let actual = process_image.read_u32(6).unwrap_or(0);
        let status = process_image.read_u16(10).unwrap_or(0);

        println!("{},{},{},0x{:04X}", step, target, actual as i32, status);
    }

    println!("\n[Master] Final bus state: {:?}", master.state);
    assert_eq!(master.state, MasterState::Running);
    eprintln!("EtherCAT servo simulation complete.");
}
