#!/bin/bash
# PreToolUse hook: block git commands that mutate the repository.
# Allows: git status, git diff, git log, git show, git branch (read-only)
# Blocks: git commit, git add, git push, git reset, git rebase, etc.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name')
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input')

if [ "$TOOL_NAME" = "runTerminalCommand" ] || [ "$TOOL_NAME" = "Bash" ]; then
  COMMAND=$(echo "$TOOL_INPUT" | jq -r '.command // .input // empty')

  if echo "$COMMAND" | grep -qE '\bgit\s+(commit|add|push|reset|rebase|merge|cherry-pick|revert|stash|tag|am|apply)\b'; then
    SAFE=$(echo "$COMMAND" | grep -qE '\bgit\s+(status|diff|log|show|branch)\b' && echo "yes")
    if [ "$SAFE" != "yes" ]; then
      echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: git mutation commands are not allowed. Use git status/diff/log only."}}'
      exit 0
    fi
  fi
fi

echo '{"continue":true}'
