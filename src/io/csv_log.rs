/// CSV data logger for control system telemetry.
///
/// Records timestamped scalar measurements to a CSV-formatted in-memory buffer.
/// Suitable for post-processing, plotting, and validation.
///
/// Only available with the `std` feature (requires allocation).
use std::string::String;
use std::vec::Vec;

/// A single CSV log row: timestamp + N channel values.
#[derive(Debug, Clone)]
pub struct LogRow<const N: usize> {
    pub time_s: f64,
    pub values: [f64; N],
}

/// CSV data logger for N channels.
pub struct CsvLog<const N: usize> {
    /// Channel names for the CSV header.
    pub channel_names: [&'static str; N],
    rows: Vec<LogRow<N>>,
    /// Maximum number of rows (0 = unlimited).
    pub max_rows: usize,
}

impl<const N: usize> CsvLog<N> {
    /// Create a new logger with channel names.
    pub fn new(channel_names: [&'static str; N]) -> Self {
        Self {
            channel_names,
            rows: Vec::new(),
            max_rows: 0,
        }
    }

    /// Create a new logger with a row capacity limit.
    pub fn with_capacity(channel_names: [&'static str; N], max_rows: usize) -> Self {
        Self {
            channel_names,
            rows: Vec::with_capacity(max_rows.min(1_000_000)),
            max_rows,
        }
    }

    /// Append a data row. Returns false if the capacity limit is reached.
    pub fn push(&mut self, time_s: f64, values: [f64; N]) -> bool {
        if self.max_rows > 0 && self.rows.len() >= self.max_rows {
            return false;
        }
        self.rows.push(LogRow { time_s, values });
        true
    }

    /// Number of rows recorded.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Access a row by index.
    pub fn row(&self, index: usize) -> Option<&LogRow<N>> {
        self.rows.get(index)
    }

    /// Render the log as a CSV string with a header row.
    ///
    /// Format: `time,ch0,ch1,...chN-1\n` followed by data rows.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("time");
        for name in &self.channel_names {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');

        for row in &self.rows {
            out.push_str(&format!("{:.6}", row.time_s));
            for &v in &row.values {
                out.push_str(&format!(",{:.6}", v));
            }
            out.push('\n');
        }
        out
    }

    /// Clear all rows.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// Iterate over rows.
    pub fn iter(&self) -> impl Iterator<Item = &LogRow<N>> {
        self.rows.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_and_rows() {
        let mut log = CsvLog::new(["pos", "vel", "acc"]);
        log.push(0.0, [1.0, 2.0, 3.0]);
        log.push(0.1, [1.1, 2.1, 3.1]);
        let csv = log.to_csv();
        assert!(csv.starts_with("time,pos,vel,acc\n"));
        assert!(csv.contains("0.000000,1.000000"));
    }

    #[test]
    fn capacity_limit() {
        let mut log = CsvLog::with_capacity(["x"], 3);
        assert!(log.push(0.0, [1.0]));
        assert!(log.push(1.0, [2.0]));
        assert!(log.push(2.0, [3.0]));
        assert!(!log.push(3.0, [4.0])); // exceeds limit
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn clear_resets_log() {
        let mut log = CsvLog::new(["v"]);
        log.push(0.0, [1.0]);
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn iter_rows() {
        let mut log = CsvLog::new(["t"]);
        log.push(0.0, [1.0]);
        log.push(0.1, [2.0]);
        let vals: Vec<f64> = log.iter().map(|r| r.values[0]).collect();
        assert_eq!(vals, vec![1.0, 2.0]);
    }
}
