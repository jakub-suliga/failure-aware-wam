#!/bin/bash
# Stop the server started by start_server.sh (via its PID file).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PID_FILE="$ROOT/logs/lingbot_server.pid"
[ -f "$PID_FILE" ] || { echo "no PID file"; exit 0; }
kill "$(cat "$PID_FILE")" 2>/dev/null && echo "stopped $(cat "$PID_FILE")"
rm -f "$PID_FILE"
