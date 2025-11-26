#!/bin/bash
# Git wrapper script to ensure all git operations are scoped to Perpetuality directory
# Usage: ./git_perpetuality.sh <git-command> [args...]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the Perpetuality project root (parent of scripts directory)
PERPETUALITY_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to Perpetuality directory
cd "$PERPETUALITY_ROOT" || {
    echo "Error: Could not change to Perpetuality directory: $PERPETUALITY_ROOT"
    exit 1
}

# Verify we're in the right git repository
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ "$GIT_ROOT" != "$PERPETUALITY_ROOT" ]; then
    echo "Error: Not in Perpetuality git repository!"
    echo "Expected: $PERPETUALITY_ROOT"
    echo "Found: $GIT_ROOT"
    exit 1
fi

# Execute git command with all arguments
git "$@"












