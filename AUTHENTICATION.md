# Authentication in CorpGrid

CorpGrid has **two distinct authentication models** for different purposes:

## 1. Consumer Authentication (API Keys)

**Purpose**: For **consuming** inference services via OpenAI-compatible APIs

**Who needs this**: Applications and users who want to run inference requests

**How it works**:
- Users create API keys via the Web UI (`/api-keys`) or admin interface
- API keys have the format `cgk_<random_32_chars>`
- Keys are hashed using bcrypt and stored in the database
- Keys can have scoped permissions and expiration dates
- Usage is tracked for billing/metrics

**Where it's used**:
- `/v1/completions` - Text completion endpoint
- `/v1/chat/completions` - Chat completion endpoint

**How to use**:
```bash
# Create an API key first (via Web UI or CLI)
# Then use it in requests:

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cgk_your_api_key_here" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Public endpoints** (no API key required):
- `/v1/models` - List available models
- `/v1/models/:id` - Get model info

---

## 2. Agent/GPU Contributor Authentication (Ed25519)

**Purpose**: For **providing** GPU resources to the cluster

**Who needs this**: Machines contributing GPU compute power (like SETI@Home or distributed.net)

**How it works**:
- Agents automatically generate Ed25519 keypairs on first run
- Keypairs are stored locally in `~/.corpgrid/agent_key`
- Public keys are registered with the scheduler on first connection
- Signatures are used for Byzantine Fault Tolerance (BFT)
- No manual setup required - agents are "open enrollment"

**Where it's used**:
- gRPC scheduler endpoints (port 50051 by default)
- Job result submission and verification
- Heartbeat/lease management

**How it works**:
```bash
# Agents automatically authenticate on startup:
corpgrid-agent --scheduler-url grpc://scheduler.example.com:50051

# On first run:
# 1. Generates Ed25519 keypair
# 2. Saves to ~/.corpgrid/agent_key (permissions 0600)
# 3. Registers public key with scheduler
# 4. Starts contributing GPU resources
```

**Why no API keys for agents?**
- Open contribution model (like distributed computing projects)
- Byzantine Fault Tolerance handles malicious actors
- Reputation system tracks reliability
- Quorum voting ensures result integrity
- Ed25519 signatures prevent result forgery

---

## Key Differences

| Aspect | Consumer Auth (API Keys) | Agent Auth (Ed25519) |
|--------|--------------------------|----------------------|
| **Purpose** | Use inference services | Provide GPU resources |
| **Authentication** | Bearer token in HTTP header | Ed25519 signature in gRPC |
| **Setup** | Manual (create API key) | Automatic (generate keypair) |
| **Permissions** | Scoped (read/write/etc.) | All agents equal |
| **Billing** | Yes (track token usage) | No (contributors donate) |
| **Security Model** | Trust-based (authorized users) | Byzantine Fault Tolerant |
| **Endpoints** | OpenAI HTTP API (port 8000) | gRPC (port 50051) |

---

## Database Schema

### Consumer Authentication Tables
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    role VARCHAR(50),  -- 'admin', 'user', 'readonly'
    ...
);

CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    key_hash VARCHAR(255),     -- bcrypt hash of API key
    key_prefix VARCHAR(20),    -- For identification (e.g., "cgk_abc123...")
    scopes TEXT[],             -- ['inference:read', 'inference:write', ...]
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    ...
);
```

### Agent Authentication Tables
```sql
CREATE TABLE devices (
    device_id VARCHAR(255) PRIMARY KEY,
    public_key BYTEA,          -- Ed25519 public key (32 bytes)
    hostname VARCHAR(255),
    gpu_count INTEGER,
    ...
);

CREATE TABLE device_reputation (
    device_id VARCHAR(255) PRIMARY KEY REFERENCES devices(device_id),
    alpha DOUBLE PRECISION,    -- Beta distribution parameter
    beta DOUBLE PRECISION,     -- Beta distribution parameter
    ...
);
```

---

## Security Considerations

### Consumer API Keys
- ✅ **Secure**: bcrypt hashed, never stored in plain text
- ✅ **Revocable**: Can be deactivated via Web UI
- ✅ **Expirable**: Optional expiration dates
- ✅ **Auditable**: All usage tracked in `audit_log` table
- ✅ **Scoped**: Fine-grained permissions per key

### Agent Ed25519 Keys
- ✅ **Cryptographically secure**: Ed25519 provides strong signatures
- ✅ **Byzantine Fault Tolerant**: Malicious results detected via quorum
- ✅ **Reputation tracking**: Beta distribution tracks reliability
- ✅ **Result verification**: All job results must be signed
- ✅ **Open participation**: No barriers to contributing resources

---

## Example Workflows

### Consumer Workflow (API Key)
1. Admin creates user account via Web UI
2. User logs in and generates API key
3. User configures application with API key
4. Application makes inference requests with `Authorization: Bearer cgk_...`
5. Usage tracked and billed per token

### Agent Workflow (Ed25519)
1. Install `corpgrid-agent` on machine with GPU
2. Run `corpgrid-agent --scheduler-url grpc://scheduler.example.com:50051`
3. Agent automatically:
   - Generates Ed25519 keypair
   - Registers with scheduler
   - Starts accepting jobs
4. Results signed with private key
5. Scheduler verifies signatures
6. Quorum ensures correctness
7. Reputation increases with successful jobs

---

## Configuration

### Enable/Disable Open Agent Enrollment

To restrict agent enrollment (optional):

```bash
# Set environment variable
export CORPGRID_AGENT_WHITELIST="device-id-1,device-id-2,device-id-3"

# Or use blacklist
export CORPGRID_AGENT_BLACKLIST="malicious-device-1,malicious-device-2"
```

### API Key Creation

Via Web UI:
1. Navigate to `/api-keys`
2. Click "Create API Key"
3. Set name, scopes, and expiration
4. Copy key immediately (shown only once)

Via CLI (TODO):
```bash
corpgrid-admin create-api-key \
  --user alice \
  --scopes inference:read,inference:write \
  --expires-in 30d
```

---

## Monitoring

### Consumer Metrics
- `/token-metrics` - Token usage per user/API key
- `/api-keys` - Active keys and last used times
- `audit_log` table - All API actions

### Agent Metrics
- `/devices` - Agent reputation scores
- `/gpu-metrics` - Real-time GPU utilization
- `device_reputation` table - Trust scores

---

## Migration from Other Systems

### From OpenAI
Your existing code works with minimal changes:

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After (CorpGrid)
from openai import OpenAI
client = OpenAI(
    api_key="cgk-...",  # CorpGrid API key
    base_url="http://your-corpgrid:8000/v1"
)
# Same API, different endpoint!
```

### From Distributed Computing Projects
Similar model to BOINC, Folding@Home:

```bash
# BOINC
boinc --attach_project http://project.url account_key

# CorpGrid
corpgrid-agent --scheduler-url grpc://scheduler.url
# No account key needed - automatic enrollment!
```

---

## Reputation-Based Trust System

### Overview

CorpGrid implements a **reputation-based trust model** for GPU contributors, eliminating the need for pre-provisioning authentication while maintaining result integrity. New agents start untrusted and gradually earn reputation through correct work.

### How It Works

**1. New Agent Registration (Untrusted State)**
```rust
// First connection:
Agent connects → Generates Ed25519 keypair → Registers with scheduler

// Initial reputation:
alpha = 1.0, beta = 1.0  // Uniform prior (50% score)
tier = "Unproven"        // < 10 samples
```

**2. Trust Tiers**

| Tier | Requirements | Replication Factor | Verification Strategy |
|------|--------------|-------------------|----------------------|
| **Unproven** | < 10 samples | base + 2 | All work verified by trusted agents |
| **Bad** | < 50% success | base + 2 | Heavy verification required |
| **Poor** | 50-70% success | base + 2 | Increased redundancy |
| **Fair** | 70-85% success | base + 1 | Standard verification |
| **Good** | 85-95% success | base | Normal redundancy |
| **Excellent** | ≥ 95% success, ≥10 samples | base - 1 | Reduced verification |

**3. Adaptive Verification**

The scheduler dynamically adjusts quorum requirements based on the reputation mix of devices working on a shard:

```rust
// Scenario 1: All trusted devices
Devices: [Good, Excellent, Fair]
Required quorum: 2 (base)

// Scenario 2: Mix of trusted and untrusted
Devices: [Unproven, Good, Unproven]
Required quorum: 2 (base)
Verification: Untrusted results must match at least one trusted device

// Scenario 3: All untrusted devices
Devices: [Unproven, Unproven, Unproven]
Required quorum: 3 (base + 1)
Verification: Require higher consensus among untrusted
```

**4. Reputation Updates**

After each job completion:
```rust
// Correct result (matches quorum consensus):
alpha += 1.0

// Incorrect result (disagrees with consensus):
beta += penalty
  - Result mismatch: +3.0
  - Timeout: +2.0
  - Heartbeat loss: +1.5
  - Checksum fail: +5.0
  - Crash: +2.5
```

**5. Trust Evolution Example**

```
Day 1 (New agent):
  Samples: 2 (alpha=1, beta=1)
  Tier: Unproven
  Strategy: Every job is verified by trusted agents

Day 7 (Proving itself):
  Samples: 15 (alpha=14, beta=1)
  Score: 93%
  Tier: Good
  Strategy: Normal redundancy, mixed with other agents

Day 30 (Established):
  Samples: 100 (alpha=96, beta=4)
  Score: 96%
  Tier: Excellent
  Strategy: Reduced redundancy, can verify untrusted agents
```

### Byzantine Fault Tolerance Integration

The reputation system enhances BFT by:

1. **Ed25519 Signatures**: Every result is cryptographically signed
   - Proves identity continuity across submissions
   - Prevents result forgery
   - Enables reputation tracking per device

2. **Quorum Voting**: Multiple agents compute the same job
   - Consensus determined by majority hash match
   - Reputation influences quorum size
   - Trusted agents act as "ground truth" for untrusted

3. **Sybil Attack Resistance**:
   - Creating many new identities doesn't help
   - All new identities start as Unproven
   - Must invest GPU time to build reputation
   - Malicious behavior tanks reputation immediately

### Security Guarantees

**Without Pre-Provisioned Auth:**
- ✅ Open enrollment (anyone can contribute)
- ✅ No manual approval needed
- ✅ Automatic trust building
- ✅ Self-healing (reputation recovers with good work)

**Against Malicious Actors:**
- ✅ New agents can't poison results (verified by trusted agents)
- ✅ Established agents lose reputation if they turn malicious
- ✅ Results are cryptographically attributable to devices
- ✅ Quorum consensus prevents single-point compromise

**Efficiency Improvements:**
- ✅ Excellent agents reduce cluster redundancy overhead
- ✅ Mixed reputation workloads optimize verification costs
- ✅ Reputation decay allows recovery from transient failures
- ✅ Wilson score confidence bounds handle statistical uncertainty

### Example: Untrusted Agent Lifecycle

```bash
# Day 1: First registration
$ corpgrid-agent --scheduler-url grpc://scheduler.example.com:50051
[INFO] Generated Ed25519 keypair
[INFO] Registered with scheduler (device_id: dev-abc123)
[INFO] Reputation tier: Unproven (0/10 samples)
[INFO] Assigned job job-001-shard-001
[INFO] Note: Your work will be verified by trusted agents

# Job completes, submits result
[INFO] Result submitted (hash: 0xabcd...)
[INFO] Quorum reached: 3/3 devices agree
[INFO] Reputation updated: alpha=2, beta=1 (67% score)

# Day 7: Building trust
[INFO] Reputation tier: Good (14/14 correct, 93% score)
[INFO] Now assigned to standard redundancy jobs

# Day 30: Trusted contributor
[INFO] Reputation tier: Excellent (96/100 correct, 96% score)
[INFO] Now helping verify untrusted agents
[INFO] Reduced redundancy applied to your jobs
```

### Database Schema

Reputation is persisted in the `device_reputation` table:

```sql
CREATE TABLE device_reputation (
    device_id TEXT PRIMARY KEY REFERENCES devices(device_id),
    alpha DOUBLE PRECISION NOT NULL DEFAULT 1.0,  -- Successes
    beta DOUBLE PRECISION NOT NULL DEFAULT 1.0,   -- Failures
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Beta distribution:
-- Score = alpha / (alpha + beta)
-- Samples = alpha + beta
-- Confidence = Wilson score interval
```

---

## FAQ

**Q: Why different auth for consumers vs. agents?**
A: Different threat models. Consumers are trusted users paying for services. Agents are untrusted contributors - Byzantine Fault Tolerance handles malicious actors.

**Q: Can agents use API keys too?**
A: No. Agents use Ed25519 for cryptographic result verification. API keys are only for consuming the inference API.

**Q: Is agent enrollment really open?**
A: Yes by default. But you can enable whitelisting/blacklisting if needed.

**Q: What prevents a malicious agent from submitting bad results?**
A: Reputation-based quorum voting. New agents (Unproven tier) have their work verified against trusted agents. As they prove reliability over ~10 jobs, they gain reputation. Malicious results are detected by consensus and immediately tank reputation.

**Q: How long does it take for a new agent to become trusted?**
A: Typically 10-15 successful jobs to reach "Good" tier (~1-2 days of normal usage). "Excellent" tier requires sustained performance over 50+ jobs (~1-2 weeks).

**Q: Can a malicious agent attack by creating many identities?**
A: No (Sybil attack resistance). All new identities start as Unproven and require GPU time investment to build reputation. Creating 100 identities just means 100 untrusted agents that must be verified by the existing trusted pool.

**Q: What if a trusted agent starts misbehaving?**
A: Reputation decays with incorrect results. A single bad result from an Excellent agent (96%+) will lower their score and tier. Multiple bad results quickly move them to Poor/Bad tier, triggering increased verification.

**Q: Can I use CorpGrid without contributing GPU resources?**
A: Yes! You can be a pure consumer with just an API key. Or pure contributor by just running an agent. Or both!
