#!/bin/bash
# Cross-node multi-communicator test
# This script handles the server/client lifecycle properly

set -e

PORT=${1:-29920}
MASTER_ADDR_MGMT="10.157.197.26"

echo "=== Multi-Comm Cross-Node Test (port=$PORT) ==="

# Clean up
pkill -f "test_mc_init.py" 2>/dev/null || true
ssh 192.10.10.226 "pkill -f 'test_mc_init.py'" 2>/dev/null || true
sleep 1

# Start server (rank 0) on 10.1
echo "Starting rank 0 (server)..."
LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib \
nohup python3 /tmp/test_mc_init.py --rank 0 --master 0.0.0.0 --port $PORT \
> /tmp/mc_server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready (check if port is listening)
for i in $(seq 1 10); do
    if ss -tlnp | grep -q ":$PORT "; then
        echo "  Server is listening (attempt $i)"
        break
    fi
    echo "  Waiting for server... (attempt $i)"
    sleep 1
done

# Start client (rank 1) on 226
echo "Starting rank 1 (client)..."
ssh 192.10.10.226 "LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib \
python3 /tmp/test_mc_init.py --rank 1 --master ${MASTER_ADDR_MGMT} --port $PORT" 2>&1
CLIENT_EXIT=$?
echo "  Client exit code: $CLIENT_EXIT"

# Give server time to complete
sleep 3

# Show server output
echo ""
echo "=== Server (rank 0) output ==="
cat /tmp/mc_server.log 2>/dev/null || echo "(no output)"

# Clean up
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Test complete ==="
