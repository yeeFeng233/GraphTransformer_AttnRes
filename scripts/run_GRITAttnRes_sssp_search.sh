#!/usr/bin/env bash
set -euo pipefail

export TASK_NAME="sssp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${SCRIPT_DIR}/run_GRITAttnRes_diam_search.sh" "$@"
