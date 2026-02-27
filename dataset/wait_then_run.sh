#!/usr/bin/env bash
# wait_then_run.sh
# Wait for a PID to exit, then execute a custom script (optionally in a
# specified directory).
#
# Usage:
#   bash wait_then_run.sh --pid <PID> --script <script_file> [--dir <dir>]
#
# Options:
#   --pid    <PID>          PID to watch; the job starts once this process exits.
#   --script <file>         Script file to execute after the PID exits.
#                           Any interpreter line (#!/usr/bin/env python3, etc.)
#                           is honoured automatically.
#   --dir    <directory>    Working directory in which to run the script.
#                           Defaults to the directory containing <script_file>.
#   --log    <file>         Append stdout/stderr to this log file in addition to
#                           the terminal.  Defaults to wait_then_run.log in <dir>.
#   --poll   <seconds>      How often to check whether the PID is still alive.
#                           Defaults to 30.
#
# Examples:
#   # Run a python script once PID 12345 finishes, from its own directory:
#   bash wait_then_run.sh --pid 12345 --script /data/jobs/run_glm.sh
#
#   # Run from a specific directory with a custom poll interval:
#   bash wait_then_run.sh --pid 12345 --script run_glm.sh --dir /data/jobs --poll 10
#
# Tip: run inside a screen/tmux session so it survives disconnects:
#   screen -S waiter bash wait_then_run.sh --pid 12345 --script /data/jobs/run_glm.sh

set -euo pipefail

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
WAIT_PID=""
JOB_SCRIPT=""
WORK_DIR=""
LOG_FILE=""
POLL_SECS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)    WAIT_PID="$2";    shift 2 ;;
        --script) JOB_SCRIPT="$2"; shift 2 ;;
        --dir)    WORK_DIR="$2";   shift 2 ;;
        --log)    LOG_FILE="$2";   shift 2 ;;
        --poll)   POLL_SECS="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --pid <PID> --script <file> [--dir <dir>] [--log <file>] [--poll <secs>]" >&2
            exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Validate required arguments
# --------------------------------------------------------------------------
if [[ -z "$WAIT_PID" ]]; then
    echo "error: --pid is required" >&2; exit 1
fi
if [[ -z "$JOB_SCRIPT" ]]; then
    echo "error: --script is required" >&2; exit 1
fi
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "error: script file not found: $JOB_SCRIPT" >&2; exit 1
fi
if ! [[ "$WAIT_PID" =~ ^[0-9]+$ ]]; then
    echo "error: --pid must be a positive integer, got: $WAIT_PID" >&2; exit 1
fi
if ! [[ "$POLL_SECS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECS" -lt 1 ]]; then
    echo "error: --poll must be a positive integer, got: $POLL_SECS" >&2; exit 1
fi

# Resolve working directory (default: directory of the script file)
if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="$(cd "$(dirname "$JOB_SCRIPT")" && pwd)"
fi
if [[ ! -d "$WORK_DIR" ]]; then
    echo "error: working directory not found: $WORK_DIR" >&2; exit 1
fi

# Resolve log file (default: wait_then_run.log inside working directory)
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="${WORK_DIR}/wait_then_run.log"
fi

# Make the script executable if it isn't already
chmod +x "$JOB_SCRIPT"

# Absolute path to the script (in case we cd away)
JOB_SCRIPT_ABS="$(cd "$(dirname "$JOB_SCRIPT")" && pwd)/$(basename "$JOB_SCRIPT")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

# --------------------------------------------------------------------------
# Wait for PID
# --------------------------------------------------------------------------
if kill -0 "$WAIT_PID" 2>/dev/null; then
    log "Waiting for PID $WAIT_PID to finish (polling every ${POLL_SECS}s)..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep "$POLL_SECS"
    done
    log "PID $WAIT_PID has exited."
else
    log "PID $WAIT_PID is not running (already finished or never existed). Proceeding immediately."
fi

# --------------------------------------------------------------------------
# Run the job script
# --------------------------------------------------------------------------
log "Starting: $JOB_SCRIPT_ABS"
log "Working directory: $WORK_DIR"
log "Log file: $LOG_FILE"

cd "$WORK_DIR"
"$JOB_SCRIPT_ABS" 2>&1 | tee -a "$LOG_FILE"

log "Job finished: $JOB_SCRIPT_ABS"
