#!/usr/bin/env bash
#
# setup.sh — start ChaksuDB's long-running services as pm2-managed,
# auto-restarting daemons:
#
#   1. chaksu-provenance-listener — consumes the 'grade_conversion' NOTIFY events
#      emitted by the auto_convert_disease_grade trigger
#      (chaksudb/ingest/framework/provenance_listener.py) and records the audit rows
#      in transformation_operations + provenance_transformations. The
#      reconcile_grade_conversions sweep (run by setup_full_database.py) is the
#      completeness backstop for any window where the listener was down.
#   2. chaksu-export-builder — the Gradio ExportSpec builder UI
#      (scripts/export_builder.py) launched with --share for a public tunnel URL.
#   3. chaksu-lab-server — the wheel + image HTTP servers (scripts/serve.py) that
#      make ChaksuDB available to other machines on the network.
#
# pm2 keeps them always-on (auto-restart on crash, restored on reboot via `pm2 save`).
#
# Idempotent: re-running reloads existing processes instead of creating duplicates.
#
# Usage:
#   ./setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- Ensure pm2 is available -------------------------------------------------
if ! command -v pm2 >/dev/null 2>&1; then
    echo "pm2 not found."
    if command -v npm >/dev/null 2>&1; then
        echo "Installing pm2 globally via npm..."
        npm install -g pm2
    else
        echo "ERROR: npm is not installed. Install Node.js/npm, then 'npm install -g pm2'." >&2
        exit 1
    fi
fi

# --- Resolve the uv launcher -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH. Install uv (https://docs.astral.sh/uv/)." >&2
    exit 1
fi
UV_BIN="$(command -v uv)"

# The listener reads DB_* config from the repo .env (via pydantic settings). pm2 runs the
# process with cwd=REPO_ROOT so the .env is picked up; warn if it's missing.
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "WARNING: no .env found at $REPO_ROOT/.env — the listener relies on DB_* env vars." >&2
fi

# --- Start or reload a pm2 process ------------------------------------------
# Use uv to run each entry inside the project's virtualenv. Args after the name
# are passed verbatim to `uv run`.
#   start_or_reload <proc_name> <uv run args...>
start_or_reload() {
    local proc_name="$1"; shift
    if pm2 describe "$proc_name" >/dev/null 2>&1; then
        echo "Reloading existing pm2 process '$proc_name'..."
        pm2 reload "$proc_name" --update-env
    else
        echo "Starting pm2 process '$proc_name'..."
        pm2 start "$UV_BIN" \
            --name "$proc_name" \
            --cwd "$REPO_ROOT" \
            --restart-delay 2000 \
            --max-restarts 1000 \
            -- "$@"
    fi
}

# 1. Grade-conversion provenance listener.
start_or_reload "chaksu-provenance-listener" \
    run python -m internal.ingest.framework.provenance_listener

# 3. Lab server (wheel + image HTTP servers).
start_or_reload "chaksu-lab-server" \
    run python scripts/serve.py

# Persist the process list so it is restored on reboot.
pm2 save

echo ""
echo "ChaksuDB services are managed by pm2:"
echo "  chaksu-provenance-listener  — grade-conversion audit listener"
echo "  chaksu-lab-server           — wheel + image servers"
echo ""
echo "  pm2 status                  # view state"
echo "  pm2 logs <name>             # tail logs (e.g. pm2 logs chaksu-provenance-listener)"
echo "  pm2 restart <name>          # manual restart"
echo ""
echo "For restart-on-boot, run the command printed by:  pm2 startup"
