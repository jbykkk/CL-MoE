# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Sync

This project uses **mutagen** for one-way file synchronization from local to a remote GPU server. Claude Code edits local files; mutagen propagates changes automatically.

- **Sync name:** `clmoe`
- **Local (Alpha):** `/Users/lyk/Documents/Projects/CL-MoE`
- **Remote (Beta):** `scut:/home/data1/lyk/Experiments/CL-MoE`
- **Sync is file-level only** — Claude Code cannot run commands on the remote server.

When providing commands that need execution (training, evaluation, installing packages, etc.), output them so the user can copy and run them on the remote server.

## Mutagen Management

```bash
# Check sync status
mutagen sync list

# Pause sync (e.g., during batch local edits to avoid partial uploads)
mutagen sync pause clmoe

# Resume sync
mutagen sync resume clmoe

# Flush pending changes immediately
mutagen sync flush clmoe
```

## Remote Server Workflow

All training and evaluation runs on the remote server (`scut`). Typical workflow:

1. Claude Code edits files locally → mutagen syncs to remote
2. User runs commands on remote server via SSH
3. Outputs/logs stay on remote; user shares relevant results back if needed

When suggesting commands, prefix with `# Run on remote server:` and use paths matching the remote layout (`/home/data1/lyk/Experiments/CL-MoE/...`).

## Conda Environment

Remote server uses conda environment `clmoe` (Python 3.10). Activate before running:

```bash
conda activate clmoe
```
