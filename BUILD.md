# Building GoGrid with GPU Support

GoGrid supports both **CUDA** (NVIDIA GPUs) and **Metal** (Apple Silicon) backends for GPU-accelerated inference. Due to platform toolchain requirements, these backends must be built separately on their respective platforms.

## Platform-Specific Build Requirements

### macOS (Apple Silicon) - Metal Backend

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- Rust nightly toolchain

**Build Command:**
```bash
cargo build --release --package corpgrid-scheduler
cargo build --release --package corpgrid-agent
```

On macOS, the binaries will automatically include Metal support via the platform-specific dependency configuration in `Cargo.toml`.

### Linux - CUDA Backend

**Requirements:**
- Linux with NVIDIA GPU
- NVIDIA CUDA Toolkit 12.2+ installed
- Rust nightly toolchain

**Build Command:**
```bash
cargo build --release --package corpgrid-scheduler
cargo build --release --package corpgrid-agent
```

On Linux, the binaries will automatically include CUDA support via the platform-specific dependency configuration in `Cargo.toml`.

#### Building with Docker (Recommended)

If you don't have direct access to a Linux machine with CUDA, you can use Docker:

**Build Docker Image:**
```bash
docker build -t gogrid-cuda-builder -f Dockerfile.cuda-builder .
```

**Build Binaries:**
```bash
mkdir -p cuda-build
docker run --rm \
  -v $(pwd):/build \
  -v $(pwd)/cuda-build:/output \
  gogrid-cuda-builder
```

The compiled `corpgrid-scheduler` binary will be in `cuda-build/`.

#### Building via SSH (Alternative)

If you have SSH access to a Linux box with CUDA:

```bash
# From macOS, sync code to Linux box
rsync -avz --exclude target --exclude .git . user@linux-box:~/GoGrid/

# SSH to Linux box
ssh user@linux-box

# Build on Linux
cd ~/GoGrid
cargo build --release --package corpgrid-scheduler
cargo build --release --package corpgrid-agent

# Copy binaries back to macOS
scp target/release/corpgrid-scheduler user@macos:~/GoGrid/target/release/corpgrid-scheduler-linux
scp target/release/corpgrid-agent user@macos:~/GoGrid/target/release/corpgrid-agent-linux
```

### Windows - CUDA Backend

**Requirements:**
- Windows with NVIDIA GPU
- NVIDIA CUDA Toolkit 12.2+ installed
- Rust nightly toolchain
- Visual Studio Build Tools

**Build Command:**
```powershell
cargo build --release --package corpgrid-scheduler
cargo build --release --package corpgrid-agent
```

On Windows, the binaries will automatically include CUDA support via the platform-specific dependency configuration in `Cargo.toml`.

## Heterogeneous Cluster Support

GoGrid supports **heterogeneous clusters** that mix CUDA and Metal devices, even though each binary supports only one GPU backend.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            Scheduler (Metal or CUDA)            │
│                                                 │
│  - Receives LoadModel requests                  │
│  - Allocates devices from registered agents     │
│  - Detects backend types from agent reports     │
│  - Routes models to appropriate agents          │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼──────────┐          ┌────────▼─────────┐
│  Agent (Metal)   │          │  Agent (CUDA)    │
│  macOS M4 Max    │          │  Linux RTX 4090  │
│  4x Metal GPUs   │          │  1x CUDA GPU     │
└──────────────────┘          └──────────────────┘
```

### How It Works

1. **Agent Registration**: Each agent reports its available GPUs and backend type (Metal or CUDA) when it registers with the scheduler

2. **Backend Detection**: The scheduler detects the backend type based on the device allocation:
   - `HomogeneousCuda`: All allocated devices are CUDA
   - `HomogeneousMetal`: All allocated devices are Metal
   - `HeterogeneousPipeline`: Mixed CUDA and Metal devices

3. **Device Selection**: When loading a model, the scheduler selects devices based on the backend type reported by agents

4. **Inference Routing**: The scheduler can route inference requests to models running on either CUDA or Metal devices

### Deployment Scenarios

#### Scenario 1: Pure Metal Cluster (macOS only)
- Build scheduler with Metal support on macOS
- Deploy to macOS machines with Apple Silicon
- All agents report Metal GPUs
- Scheduler uses `HomogeneousMetal` backend

#### Scenario 2: Pure CUDA Cluster (Linux/Windows)
- Build scheduler with CUDA support on Linux
- Deploy to Linux/Windows machines with NVIDIA GPUs
- All agents report CUDA GPUs
- Scheduler uses `HomogeneousCuda` backend

#### Scenario 3: Heterogeneous Cluster (Mixed)
- Run Metal-enabled scheduler on macOS
- Deploy Metal-enabled agents on macOS machines
- Deploy CUDA-enabled agents on Linux/Windows machines
- Agents connect to scheduler and report their GPU types
- Scheduler allocates models to appropriate agents based on backend availability

**Note**: In heterogeneous mode, each model is loaded on devices of a single backend type. The current implementation does not support splitting a single model across both CUDA and Metal devices (cross-backend pipeline parallelism). However, you can load different models on different backend types within the same cluster.

## Multi-GPU Support

GoGrid fully supports multi-GPU inference through automatic tensor parallelism:

### Homogeneous CUDA Multi-GPU (NCCL)

When multiple CUDA devices are allocated to a model:
- **Automatic Tensor Parallelism**: mistral.rs automatically enables NCCL-based tensor parallelism
- **Configuration**: Set via `CUDA_VISIBLE_DEVICES` environment variable
- **Performance**: Near-linear scaling across GPUs for large models
- **Implementation**: See `mistralrs_backend.rs:67-83`

Example: Loading a model on 4 CUDA GPUs will automatically use NCCL tensor parallelism to distribute the model across all devices.

### Homogeneous Metal Multi-GPU (Ring Backend)

When multiple Metal devices are allocated to a model:
- **Automatic Tensor Parallelism**: mistral.rs automatically enables Ring backend tensor parallelism
- **Configuration**: Automatic device detection on Apple Silicon
- **Performance**: Efficient utilization of multiple Metal GPUs (M1 Ultra, M2 Ultra, M4 Max with multiple GPUs)
- **Implementation**: See `mistralrs_backend.rs:106-116`

Example: On an M4 Max with 4 Metal GPUs, loading a model will automatically distribute it across all available GPUs.

### Heterogeneous Cross-Backend (CUDA + Metal)

When a mixed CUDA/Metal allocation is provided:
- **Primary Backend Selection**: Automatically chooses backend with most devices (ties favor CUDA)
- **Tensor Parallelism**: Applied within the primary backend only
- **Current Limitation**: Cannot split a single model across CUDA and Metal simultaneously
- **Recommendation**: Load separate model instances on each backend type for true heterogeneous clusters
- **Implementation**: See `mistralrs_backend.rs:161-219`

Example: With 2 CUDA GPUs and 1 Metal GPU, the scheduler will:
1. Select CUDA as primary backend (has more devices)
2. Load model on CUDA GPUs with NCCL tensor parallelism
3. Log that Metal GPU is available but not used for this model instance
4. Suggest loading a separate model on Metal for full utilization

## Current Limitations

1. **No Unified Binary**: Cannot build a single binary with both Metal and CUDA due to incompatible platform toolchains:
   - Metal requires macOS Objective-C compiler
   - CUDA requires NVIDIA CUDA toolkit

2. **Cross-Backend Pipeline Parallelism**: Cannot split a single model across CUDA and Metal devices due to mistral.rs architecture. Each model instance uses devices from a single backend type. For true heterogeneous inference, load separate models on each backend type.

3. **Intra-Model Cross-Backend**: While the scheduler can manage heterogeneous clusters mixing CUDA and Metal agents, individual model instances cannot span both backend types simultaneously.

## Testing

### Testing Metal Backend (macOS)
```bash
# Build and run scheduler
cargo build --release --package corpgrid-scheduler
./target/release/corpgrid-scheduler &

# Build and run agent
cargo build --release --package corpgrid-agent
./target/release/corpgrid-agent --scheduler-url grpc://localhost:50051 &

# Load model and test inference
python3 load_model.py
python3 test_inference.py <model-id>
```

### Testing CUDA Backend (Linux)
```bash
# Build and run scheduler
cargo build --release --package corpgrid-scheduler
./target/release/corpgrid-scheduler &

# Build and run agent
cargo build --release --package corpgrid-agent
./target/release/corpgrid-agent --scheduler-url grpc://localhost:50051 &

# Load model and test inference
python3 load_model.py
python3 test_inference.py <model-id>
```

### Testing Heterogeneous Cluster

1. Start Metal scheduler on macOS:
```bash
./target/release/corpgrid-scheduler
```

2. Start Metal agent on macOS:
```bash
./target/release/corpgrid-agent --scheduler-url grpc://macos-scheduler-ip:50051
```

3. Build CUDA agent on Linux:
```bash
cargo build --release --package corpgrid-agent
```

4. Start CUDA agent on Linux:
```bash
./target/release/corpgrid-agent --scheduler-url grpc://macos-scheduler-ip:50051
```

5. Load models and observe backend selection:
```bash
# The scheduler will detect and use available backends
python3 load_model.py
```

Check the scheduler logs to see which backend type was selected for each model load.

## Troubleshooting

### Metal Backend Issues

**Problem**: "No Metal devices found"
**Solution**: Ensure you're running on Apple Silicon (M1/M2/M3/M4). Intel Macs do not support Metal for ML inference.

**Problem**: Build fails with Metal framework errors
**Solution**: Install Xcode Command Line Tools: `xcode-select --install`

### CUDA Backend Issues

**Problem**: "CUDA not available" or "No CUDA devices found"
**Solution**:
- Verify CUDA toolkit is installed: `nvcc --version`
- Check NVIDIA drivers: `nvidia-smi`
- Ensure `LD_LIBRARY_PATH` includes CUDA libraries

**Problem**: Build fails with "cannot find -lcuda"
**Solution**: Set `CUDA_PATH` environment variable:
```bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Docker Build Issues

**Problem**: Docker build fails with Metal-related errors
**Solution**: This is expected. You cannot build Metal support in a Linux container. Use Docker only for CUDA builds.

**Problem**: Docker build fails with "nvcc not found"
**Solution**: Ensure you're using the correct CUDA base image: `nvidia/cuda:12.2.0-devel-ubuntu22.04`

## References

- mistral.rs documentation: https://github.com/EricLBuehler/mistral.rs
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Apple Metal: https://developer.apple.com/metal/
- Rust target-specific dependencies: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#platform-specific-dependencies
