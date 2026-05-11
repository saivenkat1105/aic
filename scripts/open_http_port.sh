#!/bin/bash

PORT=8000

cd /home/user/

echo "--- Resetting Port $PORT ---"

# 1. Find and kill the process currently using the port
PID=$(lsof -t -i :$PORT)

if [ -z "$PID" ]; then
    echo "Port $PORT is already clear."
else
    echo "Found process $PID. Closing port..."
    kill -9 $PID
    # Give the OS a split second to actually release the port
    sleep 0.5
fi

# 2. Start the Python HTTP server
echo "Starting Python3 HTTP server on port $PORT..."
# 'nohup' and '&' allow it to run in the background if you close the terminal, 
# or remove them to see the logs live.
python3 -m http.server $PORT
