#!/bin/bash
# Source this file in your .bashrc or .bash_profile to add git-perpetuality alias
# Usage: source .git_perpetuality_alias.sh
# Then use: git-perpetuality <command> [args...]

# Get the Perpetuality project root
PERPETUALITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create alias function
git-perpetuality() {
    cd "$PERPETUALITY_ROOT" || {
        echo "Error: Could not change to Perpetuality directory: $PERPETUALITY_ROOT"
        return 1
    }
    
    # Verify we're in the right git repository
    GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
    if [ "$GIT_ROOT" != "$PERPETUALITY_ROOT" ]; then
        echo "Error: Not in Perpetuality git repository!"
        echo "Expected: $PERPETUALITY_ROOT"
        echo "Found: $GIT_ROOT"
        return 1
    fi
    
    # Execute git command
    git "$@"
    
    # Return to original directory
    cd - > /dev/null
}

# Also create a safe git init function
git-init-perpetuality() {
    PERPETUALITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$PERPETUALITY_ROOT" || return 1
    
    if [ -d ".git" ]; then
        echo "Git repository already exists in $PERPETUALITY_ROOT"
        return 1
    fi
    
    git init "$@"
}

echo "Git Perpetuality aliases loaded!"
echo "Use 'git-perpetuality <command>' to run git commands in Perpetuality directory"
echo "Use 'git-init-perpetuality' to initialize git in Perpetuality directory"





