//! Communication and digital effects module.
//!
//! Provides signal-processing primitives that arise in networked and
//! digital control systems:
//!
//! - [`quantizer`]: uniform, logarithmic, and dynamic (zoom) quantizers.
//! - [`packet_dropout`]: Bernoulli and Markov packet-dropout models with ZOH.
//! - [`time_delay`]: fixed ring-buffer delay and Padé delay approximation.
#![cfg_attr(not(feature = "std"), no_std)]

pub mod packet_dropout;
pub mod quantizer;
pub mod time_delay;

pub use packet_dropout::{BernoulliDropout, DropoutError, MarkovDropout, PacketStatus};
pub use quantizer::{DynamicQuantizer, LogQuantizer, QuantizerError, UniformQuantizer};
pub use time_delay::{DelayBuffer, DelayError, PadeDelay};
