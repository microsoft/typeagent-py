# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Download Outlook emails as .eml files via Microsoft Graph API.

Requires an Azure AD app registration with Mail.Read delegated permission.
Uses ``msgraph-sdk`` and ``azure-identity`` for authentication.

Usage examples::

    # Download 50 most recent messages (interactive browser auth)
    python tools/mail/outlook_dump.py

    # Download with explicit login and app IDs
    python tools/mail/outlook_dump.py --client-id user@example.com \\
        --application-client-id 6876366c-2635-4058-ae8a-cfbe152fbd4c

    # Download 200 messages using device-code auth
    python tools/mail/outlook_dump.py --max-results 200 --device-code

    # Filter messages by sender
    python tools/mail/outlook_dump.py \\
        --filter "from/emailAddress/address eq 'alice@example.com'"

    # Full-text search (KQL)
    python tools/mail/outlook_dump.py --search "subject:quarterly report"

    # Check permissions only
    python tools/mail/outlook_dump.py --check-app-reg-permissions

    # Add Mail.Read to the app registration (requires admin)
    python tools/mail/outlook_dump.py --setup-permissions
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

from azure.identity import (
    DeviceCodeCredential,
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
)
from kiota_abstractions.base_request_configuration import RequestConfiguration
from msgraph.generated.models.o_auth2_permission_grant import (
    OAuth2PermissionGrant,
)
from msgraph.generated.users.item.messages.messages_request_builder import (
    MessagesRequestBuilder,
)
from msgraph.graph_service_client import GraphServiceClient

SCOPES = ["Mail.Read"]
ADMIN_SCOPES = ["Directory.Read.All", "DelegatedPermissionGrant.ReadWrite.All"]
OUT = Path("mail_dump")
GRAPH_RESOURCE_APP_ID = "00000003-0000-0000-c000-000000000000"  # Microsoft Graph


def _build_credential(
    args: argparse.Namespace,
) -> InteractiveBrowserCredential | DeviceCodeCredential:
    """Build an azure-identity credential with persistent token caching."""
    cache_options = TokenCachePersistenceOptions(
        name="outlook_dump",
        allow_unencrypted_storage=True,
    )
    kwargs: dict[str, object] = {
        "cache_persistence_options": cache_options,
    }
    if args.application_client_id:
        kwargs["client_id"] = args.application_client_id
    if args.tenant_id:
        kwargs["tenant_id"] = args.tenant_id
    if args.client_id:
        kwargs["login_hint"] = args.client_id

    if args.device_code:
        # Drop login_hint — DeviceCodeCredential doesn't support it.
        kwargs.pop("login_hint", None)
        return DeviceCodeCredential(**kwargs)  # type: ignore[arg-type]
    return InteractiveBrowserCredential(**kwargs)  # type: ignore[arg-type]


async def _download_messages(
    client: GraphServiceClient,
    output_dir: Path,
    max_results: int,
    odata_filter: str | None,
    search: str | None,
) -> int:
    """Download messages as .eml files. Returns the count downloaded."""
    page_size = min(max_results, 100)
    query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
        select=["id"],
        top=page_size,
        orderby=["receivedDateTime desc"],
    )
    if odata_filter:
        query_params.filter = odata_filter
    if search:
        query_params.search = f'"{search}"'

    request_config = RequestConfiguration(query_parameters=query_params)
    messages = await client.me.messages.get(request_configuration=request_config)

    count = 0
    while messages and messages.value:
        for msg in messages.value:
            if count >= max_results:
                return count
            assert msg.id is not None
            mime_bytes = await client.me.messages.by_message_id(msg.id).content.get()
            if mime_bytes:
                (output_dir / f"{msg.id}.eml").write_bytes(mime_bytes)
                count += 1
        if count >= max_results:
            break
        if messages.odata_next_link:
            messages = await client.me.messages.with_url(messages.odata_next_link).get()
        else:
            break

    return count


# ------------------------------------------------------------------
# Permission helpers
# ------------------------------------------------------------------


async def _find_service_principal(
    client: GraphServiceClient, app_id: str
) -> str | None:
    """Return the service-principal object ID for an app, or None."""
    from msgraph.generated.service_principals.service_principals_request_builder import (
        ServicePrincipalsRequestBuilder,
    )

    query = ServicePrincipalsRequestBuilder.ServicePrincipalsRequestBuilderGetQueryParameters(
        filter=f"appId eq '{app_id}'",
    )
    config = RequestConfiguration(query_parameters=query)
    result = await client.service_principals.get(request_configuration=config)
    if result and result.value:
        return result.value[0].id
    return None


async def _check_permissions(client: GraphServiceClient, app_client_id: str) -> bool:
    """Check whether Mail.Read is granted to *app_client_id*. Returns True if so."""
    from msgraph.generated.oauth2_permission_grants.oauth2_permission_grants_request_builder import (
        Oauth2PermissionGrantsRequestBuilder,
    )

    sp_id = await _find_service_principal(client, app_client_id)
    if sp_id is None:
        print(
            f"No service principal found for app {app_client_id}. "
            "Has the app been used at least once in this tenant?",
            file=sys.stderr,
        )
        return False

    query = Oauth2PermissionGrantsRequestBuilder.Oauth2PermissionGrantsRequestBuilderGetQueryParameters(
        filter=f"clientId eq '{sp_id}'",
    )
    config = RequestConfiguration(query_parameters=query)
    result = await client.oauth2_permission_grants.get(request_configuration=config)

    if result and result.value:
        for grant in result.value:
            scopes = grant.scope.split() if grant.scope else []
            if "Mail.Read" in scopes:
                print(f"Mail.Read is granted (consent type: {grant.consent_type}).")
                return True

    print("Mail.Read is NOT granted for this app registration.", file=sys.stderr)
    return False


async def _setup_permissions(client: GraphServiceClient, app_client_id: str) -> bool:
    """Grant Mail.Read to *app_client_id*. Returns True on success."""
    sp_id = await _find_service_principal(client, app_client_id)
    if sp_id is None:
        print(
            f"No service principal found for app {app_client_id}. "
            "Has the app been used at least once in this tenant?",
            file=sys.stderr,
        )
        return False

    graph_sp_id = await _find_service_principal(client, GRAPH_RESOURCE_APP_ID)
    if graph_sp_id is None:
        print(
            "Could not find Microsoft Graph service principal in this tenant.",
            file=sys.stderr,
        )
        return False

    grant = OAuth2PermissionGrant(
        client_id=sp_id,
        consent_type="AllPrincipals",
        resource_id=graph_sp_id,
        scope="Mail.Read",
    )
    await client.oauth2_permission_grants.post(grant)
    print("Mail.Read permission granted successfully.")
    return True


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Outlook emails as .eml files via Microsoft Graph API",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Login hint (email) for interactive auth (default: prompt)",
    )
    parser.add_argument(
        "--application-client-id",
        type=str,
        default=os.environ.get("OUTLOOK_APPLICATION_CLIENT_ID"),
        help=(
            "Azure AD application (client) ID "
            "(default: $OUTLOOK_APPLICATION_CLIENT_ID)"
        ),
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=os.environ.get("OUTLOOK_TENANT_ID", "common"),
        help="Azure AD tenant ID (default: $OUTLOOK_TENANT_ID or 'common')",
    )
    parser.add_argument(
        "--device-code",
        action="store_true",
        help="Use device-code auth flow instead of interactive browser",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum number of messages to download (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help=f"Output directory for .eml files (default: {OUT})",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        dest="odata_filter",
        help="OData $filter expression (e.g. \"from/emailAddress/address eq 'a@b.com'\")",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help='KQL $search query (e.g. "subject:quarterly report")',
    )
    parser.add_argument(
        "--check-app-reg-permissions",
        action="store_true",
        help="Check whether Mail.Read is granted and exit",
    )
    parser.add_argument(
        "--setup-permissions",
        action="store_true",
        help="Grant Mail.Read to the app registration (requires admin consent)",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    admin_mode = args.check_app_reg_permissions or args.setup_permissions
    scopes = SCOPES + ADMIN_SCOPES if admin_mode else SCOPES
    credential = _build_credential(args)
    client = GraphServiceClient(credentials=credential, scopes=scopes)

    app_client_id = args.application_client_id
    if args.check_app_reg_permissions:
        if not app_client_id:
            print(
                "Error: --application-client-id (or $OUTLOOK_APPLICATION_CLIENT_ID) "
                "is required for --check-app-reg-permissions.",
                file=sys.stderr,
            )
            sys.exit(1)
        ok = await _check_permissions(client, app_client_id)
        sys.exit(0 if ok else 1)

    if args.setup_permissions:
        if not app_client_id:
            print(
                "Error: --application-client-id (or $OUTLOOK_APPLICATION_CLIENT_ID) "
                "is required for --setup-permissions.",
                file=sys.stderr,
            )
            sys.exit(1)
        ok = await _setup_permissions(client, app_client_id)
        sys.exit(0 if ok else 1)

    # Normal download mode.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    count = await _download_messages(
        client,
        args.output_dir,
        args.max_results,
        args.odata_filter,
        args.search,
    )
    elapsed = time.time() - start
    print(f"Downloaded {count} messages to {args.output_dir} in {elapsed:.1f}s")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
