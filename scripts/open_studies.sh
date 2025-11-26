#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

directory=$1
declare -a pids=()

# Trap EXIT, SIGINT, SIGTERM to kill all launched processes
cleanup() {
  echo "Cleaning up..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" 2>/dev/null
      echo "Killed process $pid"
    fi
  done
  exit 0
}

trap cleanup EXIT SIGINT SIGTERM

# Launch optuna-dashboard processes
port=8800
for file in "$directory"/*.db; do
  if [ -f "$file" ]; then
    poetry run optuna-dashboard "sqlite:///$file" --port "$port" &
    pids+=($!)
    echo "Launched optuna-dashboard for $file on port $port (PID ${pids[-1]})"
    ((port++))
  fi
done

# wait a moment to ensure servers are up
sleep 2

# Open browsers for each launched process
for ((i=0; i<${#pids[@]}; i++)); do
  port=$((8800 + i))
  xdg-open "http://localhost:${port}" &
done

# Wait for all launched processes
wait "${pids[@]}"
