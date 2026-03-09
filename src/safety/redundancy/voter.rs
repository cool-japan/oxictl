use crate::core::scalar::ControlScalar;

/// Voting strategy for redundant sensor channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoterStrategy {
    /// 2-out-of-3: median of 3 channels.
    TwoOfThree,
    /// 1-out-of-2: average of 2 channels (with disagreement detection).
    OneOfTwo,
    /// Median of N channels (works for any N ≥ 1).
    Median,
}

/// Redundant sensor voter.
///
/// Compares N sensor channels and returns a voted value.
/// Detects disagreeing channels and marks them as unhealthy.
///
/// - N: number of channels
pub struct Voter<S: ControlScalar, const N: usize> {
    pub strategy: VoterStrategy,
    /// Maximum deviation from median to consider a channel healthy.
    pub tolerance: S,
    /// Health status of each channel.
    healthy: [bool; N],
}

impl<S: ControlScalar, const N: usize> Voter<S, N> {
    pub fn new(strategy: VoterStrategy, tolerance: S) -> Self {
        Self {
            strategy,
            tolerance,
            healthy: [true; N],
        }
    }

    /// Compute voted value from N channel readings.
    ///
    /// Updates channel health status.
    /// Returns the voted value, or `None` if no majority can be established.
    pub fn vote(&mut self, values: &[S; N]) -> Option<S> {
        let voted = self.compute_voted(values)?;

        // Update health: channel is healthy if within tolerance of voted value
        for (i, &v) in values.iter().enumerate().take(N) {
            self.healthy[i] = (v - voted).abs() <= self.tolerance;
        }

        Some(voted)
    }

    fn compute_voted(&self, values: &[S; N]) -> Option<S> {
        if N == 0 {
            return None;
        }
        if N == 1 {
            return Some(values[0]);
        }

        match self.strategy {
            VoterStrategy::TwoOfThree | VoterStrategy::Median => Some(median_of_n(values)),
            VoterStrategy::OneOfTwo => {
                if N < 2 {
                    return Some(values[0]);
                }
                // Average the two channels
                Some((values[0] + values[1]) * S::HALF)
            }
        }
    }

    /// Channel health status (updated by last `vote()` call).
    pub fn healthy_channels(&self) -> &[bool; N] {
        &self.healthy
    }

    /// Number of healthy channels after last vote.
    pub fn healthy_count(&self) -> usize {
        self.healthy.iter().filter(|&&h| h).count()
    }

    /// True if all channels agree (within tolerance).
    pub fn all_agree(&self) -> bool {
        self.healthy.iter().all(|&h| h)
    }
}

/// Compute the median of N values using insertion sort on a stack copy.
fn median_of_n<S: ControlScalar, const N: usize>(values: &[S; N]) -> S {
    let mut sorted = *values;
    // Insertion sort
    for i in 1..N {
        let mut j = i;
        while j > 0 && sorted[j - 1] > sorted[j] {
            sorted.swap(j - 1, j);
            j -= 1;
        }
    }
    if N % 2 == 1 {
        sorted[N / 2]
    } else {
        (sorted[N / 2 - 1] + sorted[N / 2]) * S::HALF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_of_three_median() {
        let mut voter = Voter::<f64, 3>::new(VoterStrategy::TwoOfThree, 0.5);
        // Channel 1 is faulty (high)
        let v = voter.vote(&[1.0, 1.1, 10.0]).unwrap();
        // Median of [1.0, 1.1, 10.0] = 1.1
        assert!((v - 1.1).abs() < 1e-10, "voted={}", v);
        assert!(
            !voter.healthy_channels()[2],
            "Faulty channel should be unhealthy"
        );
        assert!(voter.healthy_channels()[0]);
        assert!(voter.healthy_channels()[1]);
    }

    #[test]
    fn all_agree() {
        let mut voter = Voter::<f64, 3>::new(VoterStrategy::Median, 0.1);
        voter.vote(&[1.0, 1.05, 1.02]).unwrap();
        assert!(voter.all_agree());
    }

    #[test]
    fn one_of_two_average() {
        let mut voter = Voter::<f64, 2>::new(VoterStrategy::OneOfTwo, 0.5);
        let v = voter.vote(&[1.0, 1.2]).unwrap();
        assert!((v - 1.1).abs() < 1e-10);
    }

    #[test]
    fn single_channel() {
        let mut voter = Voter::<f64, 1>::new(VoterStrategy::Median, 0.1);
        let v = voter.vote(&[42.0]).unwrap();
        assert_eq!(v, 42.0);
    }

    #[test]
    fn healthy_count() {
        let mut voter = Voter::<f64, 4>::new(VoterStrategy::Median, 0.5);
        voter.vote(&[1.0, 1.1, 1.05, 100.0]).unwrap();
        assert_eq!(voter.healthy_count(), 3);
    }

    #[test]
    fn median_even_count() {
        let mut voter = Voter::<f64, 4>::new(VoterStrategy::Median, 1.0);
        let v = voter.vote(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        // Median of 4 = average of 2nd and 3rd = (2+3)/2 = 2.5
        assert!((v - 2.5).abs() < 1e-10);
    }
}
