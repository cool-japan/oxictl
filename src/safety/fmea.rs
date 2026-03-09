//! Failure Mode and Effect Analysis (FMEA) table with RPN calculation.
#![allow(dead_code)]

/// Severity scale 1-10 (1=no effect, 10=catastrophic).
pub type Severity = u8;
/// Occurrence scale 1-10 (1=almost never, 10=almost certain).
pub type Occurrence = u8;
/// Detection scale 1-10 (1=always detected, 10=undetectable).
pub type Detection = u8;
/// Risk Priority Number = Severity × Occurrence × Detection (max 1000).
pub type Rpn = u16;

/// Single FMEA entry.
#[derive(Debug, Clone, Copy)]
pub struct FmeaEntry {
    /// Component ID.
    pub component: u8,
    /// Failure mode ID.
    pub mode: u8,
    /// Effect ID.
    pub effect: u8,
    pub severity: Severity,
    pub occurrence: Occurrence,
    pub detection: Detection,
}

impl FmeaEntry {
    /// Create a new FMEA entry.
    pub fn new(
        component: u8,
        mode: u8,
        effect: u8,
        severity: Severity,
        occurrence: Occurrence,
        detection: Detection,
    ) -> Self {
        Self {
            component,
            mode,
            effect,
            severity,
            occurrence,
            detection,
        }
    }

    /// Compute Risk Priority Number = Severity × Occurrence × Detection.
    #[inline]
    pub fn rpn(&self) -> Rpn {
        self.severity as Rpn * self.occurrence as Rpn * self.detection as Rpn
    }
}

/// FMEA table with N slots, entries maintained in descending RPN order.
pub struct FmeaTable<const N: usize> {
    entries: [Option<FmeaEntry>; N],
    count: usize,
}

impl<const N: usize> FmeaTable<N> {
    /// Create an empty FMEA table.
    pub fn new() -> Self {
        Self {
            entries: [None; N],
            count: 0,
        }
    }

    /// Add an entry. Inserts in descending RPN order.
    /// Returns `true` if inserted, `false` if the table is full.
    pub fn add(&mut self, entry: FmeaEntry) -> bool {
        if self.count >= N {
            return false;
        }
        // Find insertion position (maintain descending RPN order).
        let rpn = entry.rpn();
        let mut pos = self.count;
        for i in 0..self.count {
            if let Some(e) = self.entries[i] {
                if rpn > e.rpn() {
                    pos = i;
                    break;
                }
            }
        }
        // Shift entries right to make room.
        let mut j = self.count;
        while j > pos {
            self.entries[j] = self.entries[j - 1];
            j -= 1;
        }
        self.entries[pos] = Some(entry);
        self.count += 1;
        true
    }

    /// Return a slice containing the top `k` entries (by RPN, descending).
    /// Returns up to `min(k, count)` entries from the sorted array.
    pub fn top_risks(&self, k: usize) -> &[Option<FmeaEntry>] {
        let limit = k.min(self.count);
        &self.entries[..limit]
    }

    /// Filter entries with severity >= `level`.
    pub fn filter_by_severity(&self, level: Severity) -> heapless::Vec<FmeaEntry, N> {
        let mut result = heapless::Vec::new();
        for slot in self.entries[..self.count].iter().flatten() {
            if slot.severity >= level {
                // Capacity is N; we have at most N entries, so push cannot fail.
                let _ = result.push(*slot);
            }
        }
        result
    }

    /// Sum of all RPNs.
    pub fn total_rpn(&self) -> u32 {
        self.entries[..self.count]
            .iter()
            .flatten()
            .map(|e| e.rpn() as u32)
            .sum()
    }

    /// Number of entries currently in the table.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<const N: usize> Default for FmeaTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpn_calculation() {
        let entry = FmeaEntry::new(1, 2, 3, 7, 5, 4);
        assert_eq!(entry.rpn(), 7 * 5 * 4);
    }

    #[test]
    fn add_sorted_by_rpn_descending() {
        let mut table: FmeaTable<8> = FmeaTable::new();
        // RPN = 6*4*3 = 72
        let e1 = FmeaEntry::new(1, 1, 1, 6, 4, 3);
        // RPN = 9*8*7 = 504
        let e2 = FmeaEntry::new(2, 2, 2, 9, 8, 7);
        // RPN = 2*2*2 = 8
        let e3 = FmeaEntry::new(3, 3, 3, 2, 2, 2);

        assert!(table.add(e1));
        assert!(table.add(e2));
        assert!(table.add(e3));
        assert_eq!(table.len(), 3);

        // Top entry should be highest RPN (504).
        let top = table.top_risks(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].unwrap().rpn(), 504);
    }

    #[test]
    fn filter_by_severity_and_total_rpn() {
        let mut table: FmeaTable<8> = FmeaTable::new();
        table.add(FmeaEntry::new(1, 1, 1, 8, 3, 2)); // sev=8, rpn=48
        table.add(FmeaEntry::new(2, 1, 1, 3, 2, 2)); // sev=3, rpn=12
        table.add(FmeaEntry::new(3, 1, 1, 9, 5, 4)); // sev=9, rpn=180

        let filtered = table.filter_by_severity(8);
        assert_eq!(filtered.len(), 2);

        // total_rpn = 48 + 12 + 180 = 240
        assert_eq!(table.total_rpn(), 240);
    }

    #[test]
    fn table_full_returns_false() {
        let mut table: FmeaTable<2> = FmeaTable::new();
        assert!(table.add(FmeaEntry::new(1, 1, 1, 1, 1, 1)));
        assert!(table.add(FmeaEntry::new(2, 2, 2, 2, 2, 2)));
        assert!(!table.add(FmeaEntry::new(3, 3, 3, 3, 3, 3)));
    }
}
