#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLMOE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Full UCIT is a single CL-MoE training pass per task. It does not reproduce
# Octopus's Stage 1 / gradient estimation / Stage 2 optimization procedure.
export UCIT_TRAIN_SAMPLING=full
export UCIT_DATA_ROOT=${UCIT_DATA_ROOT:-"$CLMOE_ROOT/data/UCIT_full"}
export UCIT_RUN_NAME=${UCIT_RUN_NAME:-full_train}

exec bash "$SCRIPT_DIR/train_all.sh" "$@"
