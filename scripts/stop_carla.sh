#!/usr/bin/env bash
# Kill the CARLA server
# Usage: bash scripts/stop_carla.sh

CARLA_PIDS=$(pgrep -f CarlaUE4-Linux)

if [ -z "$CARLA_PIDS" ]; then
    echo "No CARLA process found"
    exit 0
fi

echo "Killing CARLA processes: $CARLA_PIDS"
kill $CARLA_PIDS 2>/dev/null
sleep 2

# Force kill if still running
CARLA_PIDS=$(pgrep -f CarlaUE4-Linux)
if [ -n "$CARLA_PIDS" ]; then
    echo "Force killing: $CARLA_PIDS"
    kill -9 $CARLA_PIDS 2>/dev/null
fi

echo "CARLA stopped"
