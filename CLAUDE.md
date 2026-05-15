# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Sync

This project uses **git** to synchronize code between local and a remote GPU server.

- **Local:** `/Users/lyk/Documents/Projects/CL-MoE`
- **Remote:** `scut:/home/data1/lyk/lyk/Experiments/CL-MoE`
- **Git remote `kenny_dev`:** `https://github.com/jbykkk/CL-MoE`
- **Claude Code cannot run commands on the remote server.**

## Code Sync Workflow

1. Claude Code edits files locally → commit and push to `kenny_dev`
2. User pulls on remote server: `git -c http.sslVerify=false pull kenny_dev main`
3. Outputs/logs stay on remote; user shares relevant results back if needed

When suggesting commands, prefix with `# Run on remote server:` and use paths matching the remote layout (`/home/data1/lyk/Experiments/CL-MoE/...`).

## Remote Server Setup

Remote server uses conda environment `clmoe` (Python 3.10). Activate before running:

```bash
conda activate clmoe
export CUDA_HOME=/usr/local/cuda-12.0
export HF_HUB_OFFLINE=1
```

- `CUDA_HOME=/usr/local/cuda-12.0` — needed for DeepSpeed compilation (CUDA 11.7 doesn't support RTX 4090's compute_89)
- `HF_HUB_OFFLINE=1` — prevents HuggingFace Hub timeout issues, all model files are already local
