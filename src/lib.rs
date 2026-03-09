#![cfg_attr(not(feature = "std"), no_std)]
#![allow(rustdoc::broken_intra_doc_links)]

pub mod core;
pub mod prelude;

#[cfg(feature = "pid")]
pub mod pid;

#[cfg(feature = "safety")]
pub mod safety;

#[cfg(feature = "estimator")]
pub mod estimator;

#[cfg(feature = "state_feedback")]
pub mod state_feedback;

#[cfg(feature = "motor")]
pub mod motor;

#[cfg(feature = "scheduler")]
pub mod scheduler;

#[cfg(feature = "adaptive")]
pub mod adaptive;

#[cfg(feature = "trajectory")]
pub mod trajectory;

#[cfg(feature = "mpc")]
pub mod mpc;

#[cfg(feature = "kinematics")]
pub mod kinematics;

#[cfg(feature = "power")]
pub mod power;

#[cfg(feature = "sim")]
pub mod sim;

#[cfg(feature = "std")]
pub mod io;

#[cfg(feature = "protocol")]
pub mod protocol;

#[cfg(feature = "fuzzy")]
pub mod fuzzy;

#[cfg(feature = "optimal")]
pub mod optimal;

#[cfg(feature = "imc")]
pub mod imc;

#[cfg(feature = "neural")]
pub mod neural;

#[cfg(feature = "sysid")]
pub mod sysid;

#[cfg(feature = "flatness")]
pub mod flatness;

#[cfg(feature = "networked")]
pub mod networked;

#[cfg(feature = "geometric")]
pub mod geometric;

#[cfg(feature = "passivity")]
pub mod passivity;

#[cfg(feature = "disturbance")]
pub mod disturbance;

#[cfg(feature = "allocation")]
pub mod allocation;

#[cfg(feature = "fdi")]
pub mod fdi;

#[cfg(feature = "gp")]
pub mod gp;

#[cfg(feature = "ilc")]
pub mod ilc;

#[cfg(feature = "navigation")]
pub mod navigation;

#[cfg(feature = "extremum")]
pub mod extremum;

#[cfg(feature = "comm")]
pub mod comm;

#[cfg(feature = "optim")]
pub mod optim;

#[cfg(feature = "repetitive")]
pub mod repetitive;

#[cfg(feature = "data_driven")]
pub mod data_driven;

#[cfg(feature = "antiwindup")]
pub mod antiwindup;

#[cfg(feature = "hybrid")]
pub mod hybrid;

#[cfg(feature = "koopman")]
pub mod koopman;
