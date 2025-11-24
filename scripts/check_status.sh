#!/bin/bash
echo "=== Training Status Check ==="
echo ""

echo "1. Process Status:"
if ps aux | grep -q "[c]ompare_models.py"; then
    echo "  ✓ compare_models.py is running"
    ps aux | grep "[c]ompare_models.py" | awk '{print "    PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "  ✗ compare_models.py is NOT running"
fi

echo ""
echo "2. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not available"
fi

echo ""
echo "3. Latest Logs:"
LATEST_LOG=$(ls -t logs/*/discrete-gestures*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Latest log: $LATEST_LOG"
    echo "  Last 5 lines:"
    tail -5 "$LATEST_LOG" | sed 's/^/    /'
else
    echo "  No log files found yet"
fi

echo ""
echo "4. Results Directory:"
if [ -d "./model_comparison" ]; then
    echo "  Directory exists"
    ls -lh ./model_comparison/ 2>/dev/null | tail -5
else
    echo "  Directory not created yet"
fi

echo ""
echo "5. Training Progress (if available):"
METRICS=$(find logs/ -name "metrics.csv" -type f 2>/dev/null | head -1)
if [ -n "$METRICS" ]; then
    echo "  Latest metrics: $METRICS"
    tail -3 "$METRICS" | sed 's/^/    /'
else
    echo "  No metrics file found yet"
fi
