#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import uuid

# Azure SDK imports
from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)


def colored(text: str, color: str) -> str:
    """Add color to terminal output."""
    return f"{color}{text}{Style.RESET_ALL}"


# Get script directory and config
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "get_keys.config.json"

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DOTENV_PATH = Path(SCRIPT_DIR / config["defaultDotEnvPath"]).resolve()
SHARED_KEYS = config["env"]["shared"]
PRIVATE_KEYS = config["env"]["private"]
DELETE_KEYS = config["env"]["delete"]

AZ_COMMAND = "az.cmd" if os.name == "nt" else "az"

# Global parameters
param_shared_vault: Optional[str] = None
param_private_vault: Optional[str] = None


class AzCliLoggedInInfo:
    """Information about the Azure CLI logged-in user."""

    def __init__(self, user: str, subscription_id: str, subscription_name: str):
        self.user = user
        self.subscription_id = subscription_id
        self.subscription_name = subscription_name


async def get_az_cli_logged_in_info(print_info: bool = True) -> AzCliLoggedInInfo:
    """Get information about the Azure CLI logged-in user."""
    try:
        result = subprocess.run(
            [AZ_COMMAND, "account", "show"], capture_output=True, text=True, check=True
        )
        account = json.loads(result.stdout)

        info = AzCliLoggedInInfo(
            user=account["user"]["name"],
            subscription_id=account["id"],
            subscription_name=account["name"],
        )

        if print_info:
            print(f"Logged in as {colored(info.user, Fore.CYAN)}")
            print(
                f"Subscription: {colored(info.subscription_name, Fore.CYAN)} ({colored(info.subscription_id, Fore.CYAN)})"
            )

        return info
    except subprocess.CalledProcessError:
        raise RuntimeError("User not logged in to Azure CLI. Run 'az login'.")
    except FileNotFoundError:
        raise RuntimeError(
            "Azure CLI is not installed. Install it and run 'az login' before running this tool."
        )


class AzPIMClient:
    """Client for Azure PIM (Privileged Identity Management) operations."""

    def __init__(self, az_cli_logged_in_user: AzCliLoggedInInfo):
        self.az_cli_logged_in_user = az_cli_logged_in_user
        # fmt: off
        self.credential = DefaultAzureCredential() # CodeQL [SM05139] - This code is used as part of development setup only.
        # fmt: on

    @staticmethod
    async def create():
        """Create a new AzPIMClient instance."""
        return AzPIMClient(await get_az_cli_logged_in_info())

    async def elevate(self, options: Dict):
        """Elevate the user to a specific role using PIM."""
        subscription_id = self.az_cli_logged_in_user.subscription_id
        scope = f"/subscriptions/{subscription_id}"

        # Create authorization management client
        client = AuthorizationManagementClient(
            credential=self.credential, subscription_id=subscription_id
        )

        print(f"Looking up role information for {options['roleName']}...")

        # Get role definition ID
        role_definition_id = await self.get_role_definition_id(
            client, options["roleName"], scope
        )
        # Get principal ID
        principal_id = await self.get_principal_id(
            options.get("continueOnFailure", False)
        )

        # Create role assignment schedule request
        role_assignment_schedule_request_name = str(uuid.uuid4())
        parameters = {
            "principal_id": principal_id,
            "request_type": options["requestType"],
            "role_definition_id": role_definition_id,
            "schedule_info": {
                "expiration": {
                    "type": options["expirationType"],
                    "duration": options["expirationDuration"],
                    "end_date_time": None,
                },
                "start_date_time": options.get("startDateTime", datetime.utcnow()),
            },
            "justification": "self elevate from typeagent script",
        }

        print(
            f"Elevating {colored(principal_id, Fore.CYAN)} to {colored(options['roleName'], Fore.CYAN)} for {colored(options['expirationDuration'], Fore.CYAN)}"
        )

        try:
            result = client.role_assignment_schedule_requests.create(
                scope=scope,
                role_assignment_schedule_request_name=role_assignment_schedule_request_name,
                parameters=parameters,  # type: ignore[arg-type]
            )

            print(result)
            print(
                colored(
                    f"ELEVATION SUCCESSFUL for role {options['roleName']}", Fore.GREEN
                )
            )
        except Exception as e:
            print(repr(e))
            print(
                colored(f"Unable to elevate for role {options['roleName']}.", Fore.RED)
            )

    async def get_role_definition_id(
        self, client: AuthorizationManagementClient, role_name: str, scope: str
    ) -> str:
        """Get the role definition ID for a given role name."""
        role = None

        filter_str = "asTarget()"

        for item in client.role_eligibility_schedules.list_for_scope(
            scope=scope, filter=filter_str
        ):
            if (
                hasattr(item, "expanded_properties")
                and item.expanded_properties is not None
                and hasattr(item.expanded_properties, "role_definition")
                and item.expanded_properties.role_definition is not None
                and hasattr(item.expanded_properties.role_definition, "display_name")
                and item.expanded_properties.role_definition.display_name == role_name
            ):
                role = item
                break

        if (
            role
            and role.expanded_properties
            and role.expanded_properties.role_definition
        ):
            role_id = role.expanded_properties.role_definition.id
            if role_id:
                print(
                    f"Found Role Definition {colored(role_name, Fore.CYAN)} with id {colored(role_id, Fore.CYAN)}"
                )
                return role_id

        print(
            colored(
                f"ERROR: Unable to find the requested role '{role_name}'. Are you certain you are logged into the correct subscription?",
                Fore.RED,
            )
        )
        raise RuntimeError(f"Unable to find the role '{role_name}'.")

    async def get_principal_id(self, continue_on_failure: bool = False) -> str:
        """Get the principal ID of the current user."""
        try:
            result = subprocess.run(
                [AZ_COMMAND, "ad", "signed-in-user", "show"],
                capture_output=True,
                text=True,
                check=True,
            )
            account_details = json.loads(result.stdout)
            return account_details["id"]
        except Exception as e:
            print(repr(e))
            if not continue_on_failure:
                print(
                    colored(
                        "ERROR: Unable to get principal id of the current user.",
                        Fore.RED,
                    )
                )
                sys.exit(12)
            else:
                raise RuntimeError("Unable to get principal id of the current user.")


class AzCliKeyVaultClient:
    """Client for interacting with Azure Key Vault via Azure CLI."""

    @staticmethod
    async def create():
        """Create and validate the client."""
        try:
            result = subprocess.run(
                [AZ_COMMAND, "account", "show"],
                capture_output=True,
                text=True,
                check=True,
            )
            account = json.loads(result.stdout)
            print(f"Logged in as {colored(account['user']['name'], Fore.CYAN)}")
            return AzCliKeyVaultClient()
        except subprocess.CalledProcessError:
            print(
                colored(
                    "ERROR: User not logged in to Azure CLI. Run 'az login'.",
                    Fore.RED,
                )
            )
            sys.exit(1)
        except FileNotFoundError:
            print(
                colored(
                    "ERROR: Azure CLI is not installed. Install it and run 'az login' before running this tool.",
                    Fore.RED,
                )
            )
            sys.exit(1)

    def get_secrets(self, vault_name: str) -> List[Dict]:
        """List all secrets in a vault."""
        result = subprocess.run(
            [AZ_COMMAND, "keyvault", "secret", "list", "--vault-name", vault_name],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def read_secret(self, vault_name: str, secret_name: str) -> Dict:
        """Read a specific secret from a vault."""
        result = subprocess.run(
            [
                AZ_COMMAND,
                "keyvault",
                "secret",
                "show",
                "--vault-name",
                vault_name,
                "--name",
                secret_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def write_secret(
        self, vault_name: str, secret_name: str, secret_value: str
    ) -> Dict:
        """Write a secret to a vault."""
        result = subprocess.run(
            [
                AZ_COMMAND,
                "keyvault",
                "secret",
                "set",
                "--vault-name",
                vault_name,
                "--name",
                secret_name,
                "--value",
                secret_value,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)


async def get_secret_list_with_elevation(
    client: AzCliKeyVaultClient, vault_name: str
) -> List[Dict]:
    """Get secrets with automatic elevation if needed."""
    try:
        return client.get_secrets(vault_name)
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else str(e)
        if "ForbiddenByRbac" not in error_output:
            raise

        # Try to elevate to Key Vault Administrator
        try:
            print(colored("Elevating to get secrets...", Fore.YELLOW))
            pim_client = await AzPIMClient.create()
            await pim_client.elevate(
                {
                    "requestType": "SelfActivate",
                    "roleName": "Key Vault Administrator",
                    "expirationType": "AfterDuration",
                    "expirationDuration": "PT5M",  # 5 minutes
                    "continueOnFailure": True,
                }
            )

            print(colored("Elevation successful.", Fore.GREEN))
            print(colored("Waiting 5 seconds...", Fore.YELLOW))
            await asyncio.sleep(5)

            return client.get_secrets(vault_name)
        except Exception as e:
            print(
                colored(
                    f"Elevation to key vault admin failed...attempting to get secrets as key vault reader.\n{repr(e)}",
                    Fore.YELLOW,
                )
            )

        # Try to elevate to Key Vault Secrets User
        try:
            print(colored("Elevating to get secrets...", Fore.YELLOW))
            pim_client = await AzPIMClient.create()
            await pim_client.elevate(
                {
                    "requestType": "SelfActivate",
                    "roleName": "Key Vault Secrets User",
                    "expirationType": "AfterDuration",
                    "expirationDuration": "PT5M",  # 5 minutes
                    "continueOnFailure": True,
                }
            )

            print(colored("Elevation successful.", Fore.GREEN))
            print(colored("Waiting 5 seconds...", Fore.YELLOW))
            await asyncio.sleep(5)
        except Exception as e:
            print(
                colored(
                    f"Elevation failed...attempting to get secrets without elevation.\n{repr(e)}",
                    Fore.YELLOW,
                )
            )

        return client.get_secrets(vault_name)


async def get_secrets(
    client: AzCliKeyVaultClient, vault_name: str, shared: bool
) -> List[Tuple[str, str]]:
    """Get all enabled secrets from a vault."""
    print(
        f"Getting existing {'shared' if shared else 'private'} secrets from {colored(vault_name, Fore.CYAN)} key vault."
    )

    secret_list = await get_secret_list_with_elevation(client, vault_name)

    # Use asyncio.gather for parallel secret fetching
    async def fetch_secret(secret):
        if secret.get("attributes", {}).get("enabled", False):
            secret_name = secret["id"].split("/")[-1]
            try:
                response = client.read_secret(vault_name, secret_name)
                print(colored(f"  Found secret: {secret_name[:3]}***", Fore.GREEN))
                return (secret_name, response["value"])
            except Exception as e:
                print(
                    colored(
                        f"Failed to read secret {secret_name[:3]}***: {e}",
                        Fore.YELLOW,
                    )
                )
                return None
        return None

    results = await asyncio.gather(*[fetch_secret(secret) for secret in secret_list])
    return [r for r in results if r is not None]


def read_dotenv() -> List[Tuple[str, str]]:
    """Read the .env file."""
    if not DOTENV_PATH.exists():
        return []

    with open(DOTENV_PATH, "r") as f:
        lines = f.readlines()

    dotenv = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            dotenv.append(("", line))
            continue

        if "=" in line:
            key, *value_parts = line.split("=", 1)
            value = value_parts[0] if value_parts else ""

            if "-" in key:
                raise ValueError(
                    f"Invalid dotenv key '{key}' for key vault. Keys cannot contain dashes."
                )

            dotenv.append((key, value))
        else:
            dotenv.append(("", line))

    return dotenv


def to_secret_key(env_key: str) -> str:
    """Convert environment variable key to secret key format."""
    return env_key.replace("_", "-")


def to_env_key(secret_key: str) -> str:
    """Convert secret key to environment variable key format."""
    return secret_key.replace("-", "_")


async def push_secret(
    client: AzCliKeyVaultClient,
    vault: str,
    secrets: Dict[str, str],
    secret_key: str,
    value: str,
    shared: bool = True,
) -> int:
    """Push a secret to the vault. Returns 0=unchanged, -1=skipped, 1=updated."""
    suffix = "" if shared else " (private)"
    secret_value = secrets.get(secret_key)

    if secret_value == value:
        return 0

    if secret_key in secrets:
        print(f"  {secret_key} changed.")
        print(f"    Current value: [REDACTED]")
        print(f"    New value: [REDACTED]")
        answer = input(
            f"  Are you sure you want to overwrite the value of {secret_key}? (y/n): "
        )
        if answer.lower() != "y":
            print("Skipping...")
            return -1
        print(f"  Overwriting {secret_key}{suffix}")
    else:
        print(f"  Creating {secret_key}{suffix}")

    client.write_secret(vault, secret_key, value)
    return 1


def get_vault_names(dotenv: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Get vault names from parameters or environment."""
    return {
        "shared": param_shared_vault
        or dotenv.get("TYPEAGENT_SHAREDVAULT")
        or config["vault"]["shared"],
        "private": param_private_vault or dotenv.get("TYPEAGENT_PRIVATEVAULT"),
    }


async def push_secrets():
    """Push secrets from .env to key vault."""
    dotenv = read_dotenv()
    dotenv_dict = {k: v for k, v in dotenv if k}

    client = await AzCliKeyVaultClient.create()
    vault_names = get_vault_names(dotenv_dict)

    shared_vault = vault_names["shared"]
    assert shared_vault is not None, "Shared vault name is required"
    shared_secrets = dict(await get_secrets(client, shared_vault, True))
    private_secrets = (
        dict(await get_secrets(client, vault_names["private"], False))
        if vault_names["private"]
        else {}
    )

    print(f"Pushing secrets from {DOTENV_PATH} to key vault.")

    updated = 0
    skipped = 0

    for env_key, value in dotenv:
        if not env_key:
            continue

        secret_key = to_secret_key(env_key)

        if env_key in SHARED_KEYS:
            shared_vault = vault_names["shared"]
            assert shared_vault is not None
            result = await push_secret(
                client, shared_vault, shared_secrets, secret_key, value
            )
            if result == 1:
                updated += 1
            elif result == -1:
                skipped += 1
        elif env_key in PRIVATE_KEYS:
            private_vault = vault_names["private"]
            if not private_vault:
                print(f"  Skipping private key {env_key}.")
                continue
            result = await push_secret(
                client, private_vault, private_secrets, secret_key, value, False
            )
            if result == 1:
                updated += 1
            elif result == -1:
                skipped += 1
        else:
            print(f"  Skipping unknown key {env_key}.")

    if skipped == 0 and updated == 0:
        print("All values up to date in key vault.")
        return

    if skipped != 0:
        print(f"{skipped} secrets skipped.")
    if updated != 0:
        print(f"{updated} secrets updated.")


async def pull_secrets_from_vault(
    client: AzCliKeyVaultClient, vault_name: str, shared: bool, dotenv: Dict[str, str]
) -> Optional[int]:
    """Pull secrets from a vault."""
    keys = SHARED_KEYS if shared else PRIVATE_KEYS
    secrets = await get_secrets(client, vault_name, shared)

    if not secrets:
        print(
            colored(
                f"WARNING: No secrets found in key vault {colored(vault_name, Fore.CYAN)}.",
                Fore.YELLOW,
            )
        )
        return None

    updated = 0
    for secret_key, value in secrets:
        env_key = to_env_key(secret_key)
        if env_key in keys and dotenv.get(env_key) != value:
            print(f"  Updating {env_key[:3]}***")
            dotenv[env_key] = value
            updated += 1

    return updated


async def pull_secrets():
    """Pull secrets from key vault to .env."""
    dotenv_list = read_dotenv()
    dotenv = {k: v for k, v in dotenv_list if k}

    client = await AzCliKeyVaultClient.create()
    vault_names = get_vault_names(dotenv)

    print(f"Pulling secrets to {colored(str(DOTENV_PATH), Fore.CYAN)}")

    shared_vault = vault_names["shared"]
    assert shared_vault is not None, "Shared vault name is required"
    shared_updated = await pull_secrets_from_vault(client, shared_vault, True, dotenv)
    private_updated = None
    if vault_names["private"]:
        private_updated = await pull_secrets_from_vault(
            client, vault_names["private"], False, dotenv
        )

    if shared_updated is None and private_updated is None:
        raise RuntimeError("No secrets found in key vaults.")

    updated = (shared_updated or 0) + (private_updated or 0)

    # Delete obsolete keys
    for key in DELETE_KEYS:
        if key in dotenv:
            print(f"  Deleting {key}")
            del dotenv[key]
            updated += 1

    # Update vault names in .env
    shared_vault_name = vault_names["shared"]
    if shared_vault_name and dotenv.get("TYPEAGENT_SHAREDVAULT") != shared_vault_name:
        print("  Updating TYPEAGENT_SHAREDVAULT")
        dotenv["TYPEAGENT_SHAREDVAULT"] = shared_vault_name
        updated += 1

    if (
        vault_names["private"]
        and dotenv.get("TYPEAGENT_PRIVATEVAULT") != vault_names["private"]
    ):
        print("  Updating TYPEAGENT_PRIVATEVAULT")
        dotenv["TYPEAGENT_PRIVATEVAULT"] = vault_names["private"]
        updated += 1

    if updated == 0:
        print(f"\nAll values up to date in {colored(str(DOTENV_PATH), Fore.CYAN)}")
        return

    print(
        f"\n{updated} values updated.\nWriting '{colored(str(DOTENV_PATH), Fore.CYAN)}'."
    )

    # Write back to .env file
    with open(DOTENV_PATH, "w") as f:
        for key, value in dotenv.items():
            if key:
                f.write(f"{key}={value}\n")


def print_help():
    """Print help message."""
    print("""
Usage: get_keys.py [command] [options]

Commands:
  pull       Pull secrets from Azure Key Vault to .env file (default)
  push       Push secrets from .env file to Azure Key Vault
  help       Show this help message

Options:
  --vault VAULT_NAME      Specify shared vault name
  --private VAULT_NAME    Specify private vault name

Examples:
  python get_keys.py                           # Pull secrets (default)
  python get_keys.py pull                      # Pull secrets
  python get_keys.py push                      # Push secrets
  python get_keys.py pull --vault my-vault    # Pull from specific vault
""")


async def main():
    """Main entry point."""
    global param_shared_vault, param_private_vault

    parser = argparse.ArgumentParser(description="Manage Azure Key Vault secrets")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["push", "pull", "help"],
        default="pull",
        help="Command to execute",
    )
    parser.add_argument("--vault", help="Shared vault name")
    parser.add_argument("--private", help="Private vault name")

    args = parser.parse_args()

    param_shared_vault = args.vault
    param_private_vault = args.private

    try:
        if args.command == "push":
            await push_secrets()
        elif args.command == "pull":
            await pull_secrets()
        elif args.command == "help":
            print_help()
    except Exception as e:
        if "'az' is not recognized" in str(e) or "Azure CLI is not installed" in str(e):
            print(
                colored(
                    "ERROR: Azure CLI is not installed. Install it and run 'az login' before running this tool.",
                    Fore.RED,
                )
            )
            sys.exit(0)

        print(colored(f"FATAL ERROR: {e}", Fore.RED))
        import traceback

        traceback.print_exc()
        sys.exit(-1)


if __name__ == "__main__":
    asyncio.run(main())
