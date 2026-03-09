pub mod absolute;
pub mod endat;
pub mod hall;
pub mod incremental;
pub mod resolver;
pub mod sincos;

pub use absolute::AbsoluteEncoder;
pub use endat::{EndatDecoder, EndatResult};
pub use hall::{hall_to_sector, sector_to_angle, HallSensor};
pub use incremental::IncrementalEncoder;
pub use resolver::ResolverDecoder;
pub use sincos::SincosEncoder;
