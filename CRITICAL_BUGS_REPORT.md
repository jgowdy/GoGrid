# Critical Bugs and Defects Report

Generated: 2025-10-14

## 🔴 CRITICAL SEVERITY

### 1. Job Assignment Race Condition
**File**: `crates/scheduler/src/service.rs:159-266`
**Severity**: CRITICAL - Data corruption, duplicate work, wasted resources

**Issue**: Multiple agents polling simultaneously can be assigned the same job, causing:
- Duplicate computation and wasted resources
- Last UPDATE wins, orphaning first agent's lease
- Heartbeats from first agent fail, causing incorrect expiration handling

**Root Cause**:
- `get_pending_jobs()` fetches jobs with `status='pending'` without row-level locking
- No `SELECT ... FOR UPDATE` or similar mechanism
- Time gap between SELECT and UPDATE allows race condition

**Fix Required**: Use database row-level locking:
```sql
SELECT ... FROM job_shards WHERE status = 'pending' FOR UPDATE SKIP LOCKED LIMIT 1
```

---

### 2. Result Signature Verification Missing
**File**: `crates/scheduler/src/service.rs:374-378`
**Severity**: CRITICAL - Security vulnerability, Byzantine fault tolerance broken

**Issue**: Scheduler downloads results from S3 but never cryptographically verifies them:
```rust
// 1. Download and verify result from S3
let result_data = self.download_result_from_s3(&req.result_s3_key).await?;

// 2. Verify result signature
let result_hash = self.hash_result(&result_data);  // <-- Only hashing, not verifying!
```

**Impact**:
- Malicious agents can submit fake results with arbitrary signatures
- Agent sends `req.signature` and `req.result_hash` but neither is verified
- Breaks entire Byzantine fault tolerance model
- Quorum mechanism is meaningless if results aren't verified

**Fix Required**: Implement actual cryptographic signature verification using agent's public key

---

### 3. Sandboxed Job Arguments Not Passed
**File**: `crates/agent/src/executor.rs:74-83, 196-232`
**Severity**: CRITICAL - Jobs will fail to execute correctly

**Issue**: Job arguments are constructed but never passed to the sandboxed command:
```rust
// Line 74-81: Args are built
let args = vec![
    "--job-id".to_string(),
    assignment.job_id.clone(),
    "--shard-id".to_string(),
    assignment.shard_id.to_string(),
    ...
];

// Line 83: Args passed to run_sandboxed
self.run_sandboxed(&runner_path, &job_dir, &args).await

// Line 200: But parameter is ignored!
async fn run_sandboxed(&self, runner_path: &Path, job_dir: &Path, _args: &[String])
```

**Impact**: Job runners won't receive job_id, shard_id, or work_dir parameters

**Fix Required**: Pass args to the sandboxed Command

---

## 🟠 HIGH SEVERITY

### 4. Quorum Consensus Non-Deterministic
**File**: `crates/scheduler/src/service.rs:587-603`
**Severity**: HIGH - Breaks consensus in tie scenarios

**Issue**: When multiple result hashes tie for maximum votes, HashMap iteration is non-deterministic:
```rust
let consensus_hash = hash_counts.iter()
    .find(|(_, &count)| count == max_count)
    .map(|(hash, _)| hash.clone());
```

**Scenario**: If hash A and hash B both have 2 votes, different scheduler runs could pick different winners.

**Fix Required**: Use deterministic tie-breaker (lexicographic ordering of hashes)

---

### 5. Checkpoint Never Restored
**File**: `crates/scheduler/src/service.rs:261`
**Severity**: HIGH - Wasted computation, inefficiency

**Issue**: When reassigning expired jobs, checkpoints are never loaded:
```rust
checkpoint_s3_key: String::new(), // TODO: Check for existing checkpoints
```

**Impact**: Jobs always restart from beginning instead of resuming from checkpoint

**Fix Required**: Query `checkpoints` table for latest checkpoint before assignment

---

### 6. Consensus Hash Unsafe Unwrap
**File**: `crates/scheduler/src/service.rs:405`
**Severity**: HIGH - Potential panic

**Issue**:
```rust
self.update_reputations(&shard_results, &consensus_hash.unwrap()).await?;
```

If `check_quorum` returns `(true, None)` (shouldn't happen logically but not guaranteed), this panics.

**Fix Required**: Use `unwrap_or_else` or proper error handling

---

## 🟡 MEDIUM SEVERITY

### 7. Tar Extraction Path Traversal
**File**: `crates/agent/src/executor.rs:137-150`
**Severity**: MEDIUM - Security vulnerability

**Issue**: No protection against malicious bundles with path traversal:
```rust
Command::new("tar")
    .arg("-xzf")
    .arg(bundle_path)
    .arg("-C")
    .arg(extract_dir)
```

**Impact**: Malicious bundle could extract `../../../etc/passwd` or similar paths

**Fix Required**: Add `--no-absolute-names` flag and validate extracted paths

---

### 8. Bundle Signature Incomplete Hash
**File**: `crates/agent/src/executor.rs:153-193`
**Severity**: MEDIUM - Weak verification

**Issue**: Signature only hashes file contents, not filenames or structure:
```rust
for file in files {
    let contents = tokio::fs::read(&file).await?;
    hasher.update(&contents);  // <-- No filename included
}
```

**Impact**: Two bundles with same contents but different filenames have identical signatures

**Fix Required**: Hash (filename, content) pairs

---

### 9. Database Constraints Unknown
**File**: Database schema not visible
**Severity**: MEDIUM - Potential data integrity issues

**Issues to verify**:
- Is `attempt_id` unique?
- Are there proper foreign keys?
- Are there indexes on frequently queried columns?
- What isolation level is used for transactions?

**Fix Required**: Review and document database schema with proper constraints

---

## 🔵 LOW SEVERITY

### 10. Reputation Update Race Condition
**File**: `crates/scheduler/src/service.rs:610-619`
**Severity**: LOW - Depends on database isolation level

**Issue**: Multiple concurrent updates to same device reputation:
```rust
UPDATE device_reputation
SET alpha = alpha + $1, beta = beta + $2
WHERE device_id = $3
```

**Mitigation**: Should be safe with default READ COMMITTED isolation, but could have issues with REPEATABLE READ

---

## Summary Statistics

- **Critical Issues**: 3
- **High Severity**: 3
- **Medium Severity**: 3
- **Low Severity**: 1
- **Total**: 10

## Recommended Fix Priority

1. Fix signature verification (#2) - Security critical
2. Fix job assignment race (#1) - Correctness critical
3. Fix sandbox args (#3) - Functionality broken
4. Fix quorum tie-breaking (#4) - Consensus integrity
5. Fix checkpoint restore (#5) - Performance impact
6. Fix consensus unwrap (#6) - Stability
7. Fix tar extraction (#7) - Security
8. Fix bundle hash (#8) - Verification strength
9. Review database schema (#9) - Data integrity
10. Document isolation levels (#10) - Concurrency safety
