//! Voter (redundancy logic) validation tests.

use oxictl::safety::redundancy::dual_channel::DualChannelComparator;
use oxictl::safety::redundancy::voter::{Voter, VoterStrategy};

/// 2-of-3 vote: all agree → voted value is any of them.
#[test]
fn vote_2of3_all_agree() {
    let mut voter = Voter::<f64, 3>::new(VoterStrategy::TwoOfThree, 0.1);
    let result = voter.vote(&[5.0, 5.0, 5.0]).unwrap();
    assert!(
        (result - 5.0).abs() < 0.01,
        "All agree: voted={:.4}",
        result
    );
}

/// 2-of-3 vote: one outlier → voted from majority.
#[test]
fn vote_2of3_one_outlier() {
    let mut voter = Voter::<f64, 3>::new(VoterStrategy::TwoOfThree, 0.5);
    // Channel 2 is bad (value 10 vs 5)
    let result = voter.vote(&[5.0, 5.1, 10.0]).unwrap();
    assert!(
        (result - 5.05).abs() < 0.2,
        "Two channels agree at ~5: voted={:.4}",
        result
    );

    // Channel 2 should be marked unhealthy
    let health = voter.healthy_channels();
    assert!(health[0], "Channel 0 should be healthy");
    assert!(health[1], "Channel 1 should be healthy");
    assert!(!health[2], "Channel 2 should be unhealthy (outlier)");
}

/// 2-of-3 vote: two outliers, one good → no clear majority.
#[test]
fn vote_2of3_two_outliers() {
    let mut voter = Voter::<f64, 3>::new(VoterStrategy::TwoOfThree, 0.2);
    // Three channels all different
    let result = voter.vote(&[0.0, 5.0, 10.0]);
    // Just check we don't panic and get a result or None
    if let Some(v) = result {
        assert!(v.is_finite(), "Voted value: {:.4}", v);
    }
}

/// 1-of-2 vote: both agree → average returned.
#[test]
fn vote_1of2_both_agree() {
    let mut voter = Voter::<f64, 2>::new(VoterStrategy::OneOfTwo, 0.5);
    let result = voter.vote(&[3.0, 3.1]).unwrap();
    assert!((result - 3.05).abs() < 0.1, "voted={:.4}", result);
    assert!(voter.healthy_channels()[0]);
    assert!(voter.healthy_channels()[1]);
}

/// 1-of-2 vote: channels disagree → one marked unhealthy.
#[test]
fn vote_1of2_disagree() {
    let mut voter = Voter::<f64, 2>::new(VoterStrategy::OneOfTwo, 0.2);
    // Channels differ by 2.0 (> tolerance 0.2)
    let result = voter.vote(&[1.0, 3.0]);
    if let Some(v) = result {
        assert!(v.is_finite(), "voted={:.4}", v);
    }
    let health = voter.healthy_channels();
    let n_unhealthy = health.iter().filter(|&&h| !h).count();
    assert!(
        n_unhealthy >= 1,
        "At least one should be unhealthy on disagreement"
    );
}

/// Median voter (N=5): correct median selection.
#[test]
fn voter_median_n5() {
    let mut voter = Voter::<f64, 5>::new(VoterStrategy::Median, 0.5);
    let values = [3.0f64, 1.0, 4.0, 1.5, 2.0]; // Sorted: [1.0, 1.5, 2.0, 3.0, 4.0] → median=2.0
    let result = voter.vote(&values).unwrap();
    assert!(
        (result - 2.0).abs() < 0.01,
        "Median should be 2.0: {:.4}",
        result
    );
}

/// Voter: single channel returns that value.
#[test]
fn voter_single_channel() {
    let mut voter = Voter::<f64, 1>::new(VoterStrategy::Median, 0.1);
    let result = voter.vote(&[42.0]).unwrap();
    assert!(
        (result - 42.0).abs() < 1e-10,
        "Single channel: {:.4}",
        result
    );
}

/// Voter healthy count.
#[test]
fn voter_healthy_count_all_agree() {
    let mut voter = Voter::<f64, 3>::new(VoterStrategy::Median, 0.1);
    voter.vote(&[2.0, 2.0, 2.0]);
    assert_eq!(voter.healthy_count(), 3);
}

/// DualChannelComparator: within tolerance → OK.
#[test]
fn dual_channel_within_tolerance() {
    let mut cmp = DualChannelComparator::<f64>::new(0.5, 1);
    assert!(cmp.check(10.0, 10.3), "Within tolerance should pass");
}

/// DualChannelComparator: exceeds tolerance → fault.
#[test]
fn dual_channel_exceeds_tolerance() {
    let mut cmp = DualChannelComparator::<f64>::new(0.2, 1);
    assert!(!cmp.check(10.0, 10.5), "Exceeds tolerance should fail");
    assert!(cmp.is_tripped());
}

/// DualChannelComparator: reset clears fault.
#[test]
fn dual_channel_reset_clears_fault() {
    let mut cmp = DualChannelComparator::<f64>::new(0.1, 1);
    cmp.check(0.0, 5.0);
    assert!(cmp.is_tripped());
    cmp.reset();
    assert!(!cmp.is_tripped());
}
