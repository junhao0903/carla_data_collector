#!/usr/bin/env bash
# Start CARLA server in background and keep it running
# Usage: bash scripts/start_carla.sh

CARLA_SCRIPT=/home/adt/script/carlaUE4.sh

if [ ! -f "$CARLA_SCRIPT" ]; then
    echo "CARLA script not found: $CARLA_SCRIPT"
    exit 1
fi

# Check if CARLA is already running or ready
if pgrep -f CarlaUE4-Linux > /dev/null 2>&1; then
    echo "CARLA process already exists, waiting for port 2000..."
    for i in $(seq 1 180); do
        if ss -tlnp | grep -q ":2000"; then
            echo "CARLA is ready"
            exit 0
        fi
        sleep 1
    done
    echo "Timeout: CARLA process exists but port 2000 not ready"
    exit 1
fi

echo "Starting CARLA server..."
nohup "$CARLA_SCRIPT" -RenderOffScreen > /tmp/carla.log 2>&1 &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"
echo "Log: /tmp/carla.log"
echo "Waiting for CARLA to be ready..."

# Wait for CARLA to accept connections
for i in $(seq 1 180); do
    if ss -tlnp | grep -q ":2000"; then
        echo "CARLA is ready on port 2000"
        exit 0
    fi
    sleep 1
done

echo "Timeout: CARLA did not start within 180s"
exit 1
