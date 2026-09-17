#!/bin/bash
# Keep the same brain alive until the machine shuts down or we get SIGINT/SIGTERM.
set -u
ROOT="/Users/nick/Desktop/brian"
LOG="/Users/nick/.grok/long-running-background-tasks/brain_forever.log"
PIDFILE="/Users/nick/.brainmemory/forever.pid"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")" /Users/nick/.brainmemory
cd "$ROOT"

STOP=0
CHILD=0
trap 'STOP=1; if [ "$CHILD" -ne 0 ]; then kill -INT "$CHILD" 2>/dev/null; fi' INT TERM

echo $$ > "$PIDFILE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wrapper pid $$ starting" >> "$LOG"

while [ "$STOP" = 0 ]; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching live_brain --forever mps" >> "$LOG"
  "$PY" examples/live_brain.py --forever --device mps --web \
    --path /Users/nick/.brainmemory/live.pt \
    --ram-cap-gb 10 --ram-floor-gb 3 --pages 2 >> "$LOG" 2>&1 &
  CHILD=$!
  echo "$CHILD" > "$PIDFILE"
  wait "$CHILD"
  code=$?
  CHILD=0
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) python exit $code" >> "$LOG"
  if [ "$STOP" = 1 ]; then
    break
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) restart in 15s" >> "$LOG"
  sleep 15
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wrapper stopped" >> "$LOG"
rm -f "$PIDFILE"
