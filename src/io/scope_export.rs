/// Waveform export utilities (CSV and VCD formats).
///
/// Exports recorded waveforms from `Scope` or raw data vectors
/// into industry-standard formats for viewing in GTKWave, Python, MATLAB, etc.
///
/// Only available with the `std` feature.
use std::string::String;
use std::vec::Vec;

/// A named scalar waveform: pairs of (time_s, value).
pub struct Waveform<'a> {
    pub name: &'a str,
    pub samples: &'a [(f64, f64)],
}

/// Export multiple waveforms to CSV format.
///
/// Output: `time,name1,name2,...\n` with one row per unique timestamp.
/// If waveforms have different timestamps, missing values are left empty.
pub fn to_csv(waveforms: &[Waveform<'_>]) -> String {
    if waveforms.is_empty() {
        return String::new();
    }

    // Collect all unique timestamps sorted
    let mut times: Vec<f64> = waveforms
        .iter()
        .flat_map(|w| w.samples.iter().map(|&(t, _)| t))
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Header
    let mut out = String::from("time");
    for w in waveforms {
        out.push(',');
        out.push_str(w.name);
    }
    out.push('\n');

    // Rows — find the value for each waveform at each time
    for &t in &times {
        out.push_str(&format!("{t:.8}"));
        for w in waveforms {
            // Binary search for matching timestamp
            let val = w.samples.iter().find(|&&(ts, _)| (ts - t).abs() < 1e-12);
            if let Some(&(_, v)) = val {
                out.push_str(&format!(",{v:.8}"));
            } else {
                out.push(','); // empty
            }
        }
        out.push('\n');
    }
    out
}

/// VCD (Value Change Dump) exporter for digital and integer signals.
///
/// VCD is the standard format for logic analyzers and simulation tools (GTKWave).
/// Supports 1-bit signals at this time (extendable to bus signals).
pub struct VcdExporter {
    /// Time resolution (e.g. "1ns", "100us").
    pub timescale: &'static str,
    signals: Vec<VcdSignal>,
    events: Vec<VcdEvent>,
}

struct VcdSignal {
    name: String,
    id: char,
}

struct VcdEvent {
    time_ticks: u64,
    signal_idx: usize,
    value: bool,
}

impl VcdExporter {
    pub fn new(timescale: &'static str) -> Self {
        Self {
            timescale,
            signals: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Register a digital signal. Returns its ID character.
    pub fn add_signal(&mut self, name: &str) -> usize {
        let idx = self.signals.len();
        let id = (b'!' + idx as u8) as char; // VCD IDs start at '!'
        self.signals.push(VcdSignal {
            name: name.to_string(),
            id,
        });
        idx
    }

    /// Record a value change event.
    pub fn record(&mut self, time_ticks: u64, signal_idx: usize, value: bool) {
        self.events.push(VcdEvent {
            time_ticks,
            signal_idx,
            value,
        });
    }

    /// Generate VCD text output.
    pub fn generate(&self) -> String {
        let mut out = String::new();

        // Header
        out.push_str(&format!("$timescale {} $end\n", self.timescale));
        out.push_str("$var wire 1 ");
        for sig in &self.signals {
            out.push_str(&format!("{} {} $end\n", sig.id, sig.name));
        }
        out.push_str("$enddefinitions $end\n$dumpvars\n");

        // Initial values
        for sig in &self.signals {
            out.push_str(&format!("0{}\n", sig.id));
        }
        out.push_str("$end\n");

        // Events sorted by time
        let mut sorted_events: Vec<&VcdEvent> = self.events.iter().collect();
        sorted_events.sort_by_key(|e| e.time_ticks);

        let mut cur_time = 0u64;
        for ev in sorted_events {
            if ev.time_ticks != cur_time {
                out.push_str(&format!("#{}\n", ev.time_ticks));
                cur_time = ev.time_ticks;
            }
            let sig = &self.signals[ev.signal_idx];
            out.push_str(&format!("{}{}\n", if ev.value { 1 } else { 0 }, sig.id));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_single_waveform() {
        let samples = vec![(0.0, 1.0), (0.1, 2.0), (0.2, 3.0)];
        let wfs = [Waveform {
            name: "x",
            samples: &samples,
        }];
        let csv = to_csv(&wfs);
        assert!(csv.starts_with("time,x\n"));
        assert!(csv.contains("0.10000000,2.00000000"));
    }

    #[test]
    fn csv_multiple_waveforms() {
        let s1 = vec![(0.0, 1.0), (1.0, 2.0)];
        let s2 = vec![(0.0, 3.0), (1.0, 4.0)];
        let wfs = [
            Waveform {
                name: "a",
                samples: &s1,
            },
            Waveform {
                name: "b",
                samples: &s2,
            },
        ];
        let csv = to_csv(&wfs);
        assert!(csv.starts_with("time,a,b\n"));
    }

    #[test]
    fn vcd_basic() {
        let mut vcd = VcdExporter::new("1us");
        let clk = vcd.add_signal("clk");
        vcd.record(0, clk, false);
        vcd.record(5, clk, true);
        vcd.record(10, clk, false);

        let out = vcd.generate();
        assert!(out.contains("$timescale 1us $end"));
        assert!(out.contains("clk"));
        assert!(out.contains("#5"));
        assert!(out.contains("1!"));
    }

    #[test]
    fn empty_waveforms_csv() {
        let csv = to_csv(&[]);
        assert!(csv.is_empty());
    }
}
