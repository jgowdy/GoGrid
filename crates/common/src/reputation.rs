use serde::{Deserialize, Serialize};

/// Device reputation using Beta distribution
/// Tracks reliability through successes (alpha) and failures (beta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceReputation {
    /// Successes (α parameter)
    pub alpha: f64,
    /// Failures (β parameter)
    pub beta: f64,
    /// Decay factor to allow recovery (0.0 to 1.0)
    pub decay_rate: f64,
}

impl DeviceReputation {
    /// Create new reputation with prior
    pub fn new() -> Self {
        Self {
            alpha: 1.0, // Uniform prior
            beta: 1.0,
            decay_rate: 0.01, // 1% decay per update cycle
        }
    }

    /// Create with custom prior
    pub fn with_prior(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            decay_rate: 0.01,
        }
    }

    /// Record a successful job completion
    pub fn record_success(&mut self) {
        self.alpha += 1.0;
        self.apply_decay();
    }

    /// Record a failure (timeout, mismatch, etc.)
    pub fn record_failure(&mut self, penalty: f64) {
        self.beta += penalty;
        self.apply_decay();
    }

    /// Apply gradual decay toward prior
    fn apply_decay(&mut self) {
        let total = self.alpha + self.beta;
        if total > 2.0 {
            // Decay toward uniform prior (1, 1)
            self.alpha = self.alpha * (1.0 - self.decay_rate) + self.decay_rate;
            self.beta = self.beta * (1.0 - self.decay_rate) + self.decay_rate;
        }
    }

    /// Get reputation score (mean of Beta distribution)
    pub fn score(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get lower confidence bound (for optimistic selection)
    /// Uses Wilson score with 95% confidence
    pub fn lower_bound(&self) -> f64 {
        let n = self.alpha + self.beta;
        let p = self.score();

        // Wilson score interval
        let z = 1.96; // 95% confidence
        let denominator = 1.0 + z * z / n;
        let centre = p + z * z / (2.0 * n);
        let adjustment = z * ((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))).sqrt();

        ((centre - adjustment) / denominator).max(0.0)
    }

    /// Get reputation tier for scheduling decisions
    pub fn tier(&self) -> ReputationTier {
        let score = self.score();
        let samples = self.alpha + self.beta;

        // Need minimum samples for high tier
        if samples < 10.0 {
            return ReputationTier::Unproven;
        }

        if score >= 0.95 {
            ReputationTier::Excellent
        } else if score >= 0.85 {
            ReputationTier::Good
        } else if score >= 0.70 {
            ReputationTier::Fair
        } else if score >= 0.50 {
            ReputationTier::Poor
        } else {
            ReputationTier::Bad
        }
    }

    /// Determine replication factor based on reputation
    pub fn required_replication(&self, base_factor: u32) -> u32 {
        match self.tier() {
            ReputationTier::Excellent => base_factor.saturating_sub(1).max(1),
            ReputationTier::Good => base_factor,
            ReputationTier::Fair => base_factor + 1,
            ReputationTier::Poor => base_factor + 2,
            ReputationTier::Bad | ReputationTier::Unproven => base_factor + 2,
        }
    }

    /// Determine heartbeat timeout multiplier
    pub fn timeout_multiplier(&self) -> f64 {
        match self.tier() {
            ReputationTier::Excellent => 1.5,
            ReputationTier::Good => 1.0,
            ReputationTier::Fair => 0.8,
            ReputationTier::Poor => 0.6,
            ReputationTier::Bad | ReputationTier::Unproven => 0.5,
        }
    }
}

impl Default for DeviceReputation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReputationTier {
    Excellent, // ≥95% success, ≥10 samples
    Good,      // ≥85%
    Fair,      // ≥70%
    Poor,      // ≥50%
    Bad,       // <50%
    Unproven,  // <10 samples
}

/// Failure penalty weights
pub struct FailurePenalty;

impl FailurePenalty {
    pub const TIMEOUT: f64 = 2.0;
    pub const RESULT_MISMATCH: f64 = 3.0;
    pub const HEARTBEAT_LOSS: f64 = 1.5;
    pub const CHECKSUM_FAIL: f64 = 5.0;
    pub const CRASH: f64 = 2.5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_new() {
        let rep = DeviceReputation::new();
        assert_eq!(rep.score(), 0.5); // Uniform prior
        assert_eq!(rep.tier(), ReputationTier::Unproven);
    }

    #[test]
    fn test_reputation_success() {
        let mut rep = DeviceReputation::new();

        // Record 10 successes
        for _ in 0..10 {
            rep.record_success();
        }

        assert!(rep.score() > 0.9);
        assert_eq!(rep.tier(), ReputationTier::Excellent);
    }

    #[test]
    fn test_reputation_failure() {
        let mut rep = DeviceReputation::new();

        // Record mixed results
        for _ in 0..5 {
            rep.record_success();
        }
        for _ in 0..5 {
            rep.record_failure(FailurePenalty::TIMEOUT);
        }

        let score = rep.score();
        assert!(score < 0.6); // Should be lower due to failures
    }

    #[test]
    fn test_reputation_tiers() {
        let mut rep = DeviceReputation::new();

        // Build up excellent reputation
        for _ in 0..20 {
            rep.record_success();
        }
        assert_eq!(rep.tier(), ReputationTier::Excellent);

        // Add some failures
        for _ in 0..5 {
            rep.record_failure(FailurePenalty::TIMEOUT);
        }
        assert!(rep.tier() != ReputationTier::Excellent);
    }

    #[test]
    fn test_replication_factor() {
        let mut rep = DeviceReputation::new();

        // Excellent reputation reduces replication
        for _ in 0..20 {
            rep.record_success();
        }
        assert_eq!(rep.required_replication(2), 1);

        // Poor reputation increases replication
        let mut poor_rep = DeviceReputation::new();
        for _ in 0..10 {
            poor_rep.record_failure(FailurePenalty::RESULT_MISMATCH);
        }
        assert!(poor_rep.required_replication(2) > 2);
    }

    #[test]
    fn test_lower_bound() {
        let mut rep = DeviceReputation::new();
        for _ in 0..100 {
            rep.record_success();
        }

        let score = rep.score();
        let lb = rep.lower_bound();

        assert!(lb <= score);
        assert!(lb > 0.0);
    }
}
