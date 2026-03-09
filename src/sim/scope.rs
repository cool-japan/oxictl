/// Simple waveform recorder for simulation analysis.
/// Records (time, value) pairs.
pub struct Scope {
    data: Vec<(f64, f64)>,
    name: &'static str,
}

impl Scope {
    pub fn new(name: &'static str) -> Self {
        Self {
            data: Vec::new(),
            name,
        }
    }

    pub fn with_capacity(name: &'static str, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            name,
        }
    }

    /// Record a data point.
    pub fn record(&mut self, time: f64, value: f64) {
        self.data.push((time, value));
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn data(&self) -> &[(f64, f64)] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the last recorded value.
    pub fn last_value(&self) -> Option<f64> {
        self.data.last().map(|&(_, v)| v)
    }

    /// Generate CSV string: "time,{name}\n..."
    pub fn to_csv(&self) -> String {
        let mut csv = format!("time,{}\n", self.name);
        for &(t, v) in &self.data {
            csv.push_str(&format!("{},{}\n", t, v));
        }
        csv
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Find the maximum value.
    pub fn max_value(&self) -> Option<f64> {
        self.data.iter().map(|&(_, v)| v).reduce(f64::max)
    }

    /// Find the minimum value.
    pub fn min_value(&self) -> Option<f64> {
        self.data.iter().map(|&(_, v)| v).reduce(f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_retrieve() {
        let mut scope = Scope::new("test");
        scope.record(0.0, 1.0);
        scope.record(0.1, 2.0);
        assert_eq!(scope.len(), 2);
        assert_eq!(scope.data()[0], (0.0, 1.0));
        assert_eq!(scope.data()[1], (0.1, 2.0));
    }

    #[test]
    fn last_value() {
        let mut scope = Scope::new("test");
        assert_eq!(scope.last_value(), None);
        scope.record(0.0, 42.0);
        assert_eq!(scope.last_value(), Some(42.0));
    }

    #[test]
    fn csv_output() {
        let mut scope = Scope::new("temperature");
        scope.record(0.0, 25.0);
        scope.record(0.1, 26.0);
        let csv = scope.to_csv();
        assert!(csv.starts_with("time,temperature\n"));
        assert!(csv.contains("0,25\n"));
    }

    #[test]
    fn min_max() {
        let mut scope = Scope::new("test");
        scope.record(0.0, 5.0);
        scope.record(0.1, 2.0);
        scope.record(0.2, 8.0);
        assert_eq!(scope.max_value(), Some(8.0));
        assert_eq!(scope.min_value(), Some(2.0));
    }

    #[test]
    fn clear_empties() {
        let mut scope = Scope::new("test");
        scope.record(0.0, 1.0);
        scope.clear();
        assert!(scope.is_empty());
    }
}
