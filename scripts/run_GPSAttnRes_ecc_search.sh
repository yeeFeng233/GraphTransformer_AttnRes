#!/usr/bin/env bash
set -euo pipefail

export TASK_NAME="ecc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${SCRIPT_DIR}/run_GPSAttnRes_diam_search.sh" "$@"
