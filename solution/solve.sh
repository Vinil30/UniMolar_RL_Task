#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${TASK_DIR:-/app}"
PYTHON_BIN="${PYTHON:-python}"
export TASK_DIR

RESULT_DIR="${OTTER_RESULTS_DIR:-$TASK_DIR/results}"
mkdir -p "$RESULT_DIR"

printf '== invariant molecular task runner ==\n'
printf 'WORKDIR=%s\n' "$(cd "$SCRIPT_DIR/.." && pwd)"
printf 'INPUTS=%s\n' "${OTTER_INPUTS_DIR:-$TASK_DIR/task_inputs}"
echo "SOLUTION=$SCRIPT_DIR"

printf '=== launching training and artifact export ===\n'
RUN_LOG="$RESULT_DIR/run.log"
set +e
"$PYTHON_BIN" "$SCRIPT_DIR/solution.py" >"$RUN_LOG" 2>&1
status=$?
set -e
sed -n '1,200000p' "$RUN_LOG"
if [[ $status -ne 0 ]]; then
  exit "$status"
fi

printf '=== artifacts written ===\n'
