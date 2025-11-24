#!/bin/bash
# Quick status check for the comparison script

echo "=== Comparison Script Status Check ==="
echo ""

# Find the latest training log
LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training log files found"
    exit 1
fi

echo "📄 Latest log: $LATEST_LOG"
echo ""

# Check if process is running
PID=$(ps aux | grep "compare_models.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$PID" ]; then
    echo "✅ Comparison script is RUNNING (PID: $PID)"
    echo ""
    echo "Process details:"
    ps -p $PID -o pid,etime,pcpu,pmem,cmd 2>/dev/null | tail -1
    echo ""
else
    echo "❌ Comparison script is NOT running"
    echo ""
fi

# Check log file activity
echo "📊 Log file status:"
FILE_SIZE=$(stat -c "%s" "$LATEST_LOG" 2>/dev/null)
FILE_TIME=$(stat -c "%y" "$LATEST_LOG" 2>/dev/null | cut -d'.' -f1)
echo "  Size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "${FILE_SIZE} bytes")"
echo "  Last modified: $FILE_TIME"
echo ""

# Show last 10 lines of log
echo "📝 Last 10 lines of log:"
echo "---"
tail -10 "$LATEST_LOG" 2>/dev/null | sed 's/^/  /'
echo "---"
echo ""

# Check GPU usage
echo "🎮 GPU Usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name util mem_used mem_total; do
    echo "  GPU $idx: $util | Memory: $mem_used / $mem_total"
done

echo ""
echo "💡 To monitor in real-time:"
echo "  tail -f $LATEST_LOG"

