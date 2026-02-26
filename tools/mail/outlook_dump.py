# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Download Outlook emails as .eml files via Microsoft Graph API.

Requires an Azure AD app registration with Mail.Read delegated permission.
Uses ``msgraph-sdk`` and ``azure-identity`` for authentication.

Usage examples::

    # Download 50 most recent messages (interactive browser auth)
    python tools/mail/outlook_dump.py

    # Download with explicit login and app IDs
    python tools/mail/outlook_dump.py --client-id user@example.com \
        --application-client-id 6876366c-2635-4058-ae8a-cfbe152fbd4c

    # Download 200 messages using device-code auth
    python tools/mail/outlook_dump.py --max-results 200 --device-code

    # Filter messages by sender
    python tools/mail/outlook_dump.py \
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
import re
import time
from uuid import UUID

from azure.identity import (
    DefaultAzureCredential,
    DeviceCodeCredential,
    InteractiveBrowserCredential,
)
from kiota_abstractions.base_request_configuration import RequestConfiguration
from msgraph.generated.applications.applications_request_builder import (
    ApplicationsRequestBuilder,
)
from msgraph.generated.models.application import Application
from msgraph.generated.models.o_data_errors.o_data_error import ODataError
from msgraph.generated.models.required_resource_access import RequiredResourceAccess
from msgraph.generated.models.resource_access import ResourceAccess
from msgraph.generated.users.item.messages.messages_request_builder import (  # type: ignore[import-not-found]
    MessagesRequestBuilder,
)
from msgraph.graph_service_client import GraphServiceClient

# Delegated scopes requested at sign-in
REQUIRED_SCOPES = ["Mail.Read"]

# Required delegated permissions to check with --check-app-reg-permissions
REQUIRED_DELEGATED_PERMISSIONS = [
    "Mail.Read",
    "User.Read",
]

# Default output directory
OUT = Path("mail_dump")

# Well-known Microsoft Graph application ID
GRAPH_APP_ID = "00000003-0000-0000-c000-000000000000"
# Mail.Read *delegated* permission GUID
MAIL_READ_SCOPE_ID = "570282fd-fa5c-430d-a7fd-fc8dc98a9dca"

type Credential = DefaultAzureCredential | InteractiveBrowserCredential | DeviceCodeCredential


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_credential(
    application_client_id: str | None,
    tenant_id: str | None,
    use_device_code: bool,
    login_hint: str | None = None,
) -> Credential:
    """Create an Azure credential for authentication.

    When *application_client_id* is provided, uses interactive browser or
    device-code authentication against that specific app registration.

    When *application_client_id* is ``None``, falls back to
    ``DefaultAzureCredential`` which tries Azure CLI, VS Code, managed
    identity, and other ambient credentials automatically.
    """
    if not application_client_id:
        print("No --application-client-id specified; using DefaultAzureCredential.")
        return DefaultAzureCredential()
    if use_device_code:
        return DeviceCodeCredential(
            client_id=application_client_id, tenant_id=tenant_id
        )
    return InteractiveBrowserCredential(
        client_id=application_client_id,
        tenant_id=tenant_id,
        login_hint=login_hint,
    )


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


async def check_permissions(client: GraphServiceClient) -> bool:
    """Verify that the authenticated user has Mail.Read access.

    Makes a minimal ``GET /me/messages?$top=1&$select=id`` call.
    Returns ``True`` when the call succeeds, ``False`` otherwise.
    """
    query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
        top=1,
        select=["id"],
    )
    config = RequestConfiguration(query_parameters=query_params)
    try:
        await client.me.messages.get(request_configuration=config)
    except ODataError as e:
        code = e.error.code if e.error else "Unknown"
        message = e.error.message if e.error else str(e)
        print(f"Permission check failed ({code}): {message}")
        print("Ensure the app has Mail.Read and the user has consented.")
        return False
    print("Mail.Read permission verified successfully.")
    return True


async def check_app_registration_permissions(
    credential: Credential, application_client_id: str | None
) -> bool:
    """Check whether Mail.Read is present in the token's granted scopes.

    Acquires an access token for the Microsoft Graph ``Mail.Read`` scope and
    decodes the JWT (without cryptographic verification) to inspect:

    * ``scp`` – delegated scopes actually granted to the token.
    * ``appid`` / ``azp`` – confirms the correct application client ID.
    * ``tid`` – tenant the token was issued for.

    No admin permissions are required; the token is obtained with the same
    credential used for all other operations.  Returns ``True`` when
    ``Mail.Read`` appears in the granted scopes, ``False`` otherwise.
    """
    import base64
    import json

    if not application_client_id:
        print("Cannot inspect permissions without --application-client-id.")
        return False

    # Acquire a token for the Graph Mail.Read scope
    try:
        token = credential.get_token("https://graph.microsoft.com/Mail.Read")
    except Exception as e:
        print(f"Failed to acquire token: {e}")
        print("Ensure the app registration has Mail.Read configured and consented.")
        return False

    # Decode the JWT payload (no signature verification needed here)
    parts = token.token.split(".")
    if len(parts) < 2:
        print("Access token is not a valid JWT.")
        return False

    payload_b64 = parts[1]
    # Fix base64 padding
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to decode token payload: {e}")
        return False

    # Display token metadata
    app_id = payload.get("appid") or payload.get("azp") or "unknown"
    tenant_id = payload.get("tid", "unknown")
    upn = payload.get("upn") or payload.get("preferred_username") or "unknown"
    print(f"Token info:")
    print(f"  Application client ID: {app_id}")
    print(f"  Tenant ID:             {tenant_id}")
    print(f"  User:                  {upn}")

    if app_id != application_client_id:
        print(
            f"  WARNING: token appid '{app_id}' does not match"
            f" --application-client-id '{application_client_id}'"
        )

    # Check granted scopes
    granted_scopes = set((payload.get("scp") or "").split())
    print(f"  Granted scopes:        {' '.join(sorted(granted_scopes)) or '(none)'}")

    all_ok = True
    for scope in REQUIRED_DELEGATED_PERMISSIONS:
        if scope in granted_scopes:
            print(f"  {scope}: GRANTED")
        else:
            print(f"  {scope}: NOT GRANTED")
            all_ok = False

    if not all_ok:
        missing = sorted(set(REQUIRED_DELEGATED_PERMISSIONS) - granted_scopes)
        print(f"\n  Missing permissions: {', '.join(missing)}")
        print("  The permissions may not be configured or admin consent is required.")
        print("  Use --setup-permissions to add them, or visit:")
        print(
            f"    https://login.microsoftonline.com/common/adminconsent"
            f"?client_id={application_client_id}"
        )

    # ------------------------------------------------------------------
    # Check redirect URI configuration on the app registration
    # ------------------------------------------------------------------
    redirect_ok = await _check_redirect_uri(credential, application_client_id)
    if not redirect_ok:
        all_ok = False

    return all_ok


async def _check_redirect_uri(
    credential: Credential, application_client_id: str
) -> bool:
    """Verify the app registration has http://localhost as a public client redirect URI.

    Queries ``GET /applications?$filter=appId eq '...'`` and inspects
    ``publicClient.redirectUris``.  Requires ``Application.Read.All`` or
    ownership of the app registration.  If the query fails due to
    insufficient permissions, prints a warning and returns ``True``
    (optimistic — the caller cannot determine the answer).
    """
    EXPECTED_REDIRECT_URI = "http://localhost"

    client = GraphServiceClient(credential, ["Application.Read.All"])

    app_params = (
        ApplicationsRequestBuilder.ApplicationsRequestBuilderGetQueryParameters(
            filter=f"appId eq '{application_client_id}'",
            select=["id", "appId", "displayName", "publicClient", "web", "spa"],
        )
    )
    app_config = RequestConfiguration(query_parameters=app_params)

    try:
        apps_response = await client.applications.get(request_configuration=app_config)
    except ODataError as e:
        code = e.error.code if e.error else "Unknown"
        message = e.error.message if e.error else str(e)
        print(f"\nRedirect URI check skipped ({code}): {message}")
        print("  This check requires Application.Read.All or app ownership.")
        return True  # optimistic — cannot verify

    if not apps_response or not apps_response.value:
        print(
            f"\nRedirect URI check: app registration not found ({application_client_id})"
        )
        return False

    app = apps_response.value[0]

    # "Mobile and desktop applications" → publicClient.redirectUris
    public_uris: list[str] = []
    if app.public_client and app.public_client.redirect_uris:
        public_uris = list(app.public_client.redirect_uris)

    # Also show web and SPA redirect URIs for context
    web_uris: list[str] = []
    if app.web and app.web.redirect_uris:
        web_uris = list(app.web.redirect_uris)

    spa_uris: list[str] = []
    if app.spa and app.spa.redirect_uris:
        spa_uris = list(app.spa.redirect_uris)

    print("\nRedirect URI configuration:")
    if public_uris:
        print(f"  Mobile and desktop (publicClient): {', '.join(public_uris)}")
    else:
        print("  Mobile and desktop (publicClient): (none)")
    if web_uris:
        print(f"  Web:                               {', '.join(web_uris)}")
    if spa_uris:
        print(f"  SPA:                               {', '.join(spa_uris)}")

    if EXPECTED_REDIRECT_URI in public_uris:
        print(f"  {EXPECTED_REDIRECT_URI} in publicClient: OK")
        return True

    print(f"  {EXPECTED_REDIRECT_URI} in publicClient: NOT FOUND")
    print("  To fix, go to Azure Portal > Microsoft Entra ID > App registrations")
    print(f"  > {application_client_id} > Authentication")
    print("  > Add a platform > Mobile and desktop applications")
    print(f"  > check '{EXPECTED_REDIRECT_URI}' > Save")
    return False


async def setup_permissions(
    credential: Credential, application_client_id: str | None
) -> bool:
    """Add the Mail.Read delegated permission to an app registration.

    Requires ``Application.ReadWrite.All`` or the Global Administrator role.
    Returns ``True`` when the permission is already present or was added
    successfully, ``False`` on failure.
    """
    if not application_client_id:
        print("--application-client-id is required for --setup-permissions.")
        return False

    admin_scopes = ["Application.ReadWrite.All"]
    admin_client = GraphServiceClient(credential, admin_scopes)

    try:
        # Locate the application object by its client (app) ID
        query_params = (
            ApplicationsRequestBuilder.ApplicationsRequestBuilderGetQueryParameters(
                filter=f"appId eq '{application_client_id}'",
            )
        )
        config = RequestConfiguration(query_parameters=query_params)
        apps_response = await admin_client.applications.get(
            request_configuration=config
        )

        if not apps_response or not apps_response.value:
            print(
                f"No application found with application_client_id: {application_client_id}"
            )
            _print_manual_setup_instructions(application_client_id)
            return False

        app = apps_response.value[0]
        app_object_id = app.id
        if not app_object_id:
            print("Application object has no ID.")
            return False
        existing_access = list(app.required_resource_access or [])

        # Find or create the Microsoft Graph resource entry
        graph_resource: RequiredResourceAccess | None = None
        for resource in existing_access:
            if resource.resource_app_id == GRAPH_APP_ID:
                graph_resource = resource
                break

        # Check whether Mail.Read is already configured
        if graph_resource:
            for access in graph_resource.resource_access or []:
                if str(access.id) == MAIL_READ_SCOPE_ID:
                    print("Mail.Read permission is already configured.")
                    return True

        # Build the new permission entry
        mail_read = ResourceAccess()
        mail_read.id = UUID(MAIL_READ_SCOPE_ID)
        mail_read.type = "Scope"

        if graph_resource:
            graph_resource.resource_access = list(
                graph_resource.resource_access or []
            ) + [mail_read]
        else:
            graph_resource = RequiredResourceAccess()
            graph_resource.resource_app_id = GRAPH_APP_ID
            graph_resource.resource_access = [mail_read]
            existing_access.append(graph_resource)

        # Patch the application
        update_body = Application()
        update_body.required_resource_access = existing_access
        await admin_client.applications.by_application_id(app_object_id).patch(
            update_body
        )

        print("Mail.Read permission added to app registration.")
        print("Admin consent may still be required.  Visit:")
        print(
            f"  https://login.microsoftonline.com/common/adminconsent"
            f"?client_id={application_client_id}"
        )
        return True

    except ODataError as e:
        code = e.error.code if e.error else "Unknown"
        message = e.error.message if e.error else str(e)
        print(f"Failed to set up permissions ({code}): {message}")
        _print_manual_setup_instructions(application_client_id)
        return False


def _print_manual_setup_instructions(client_id: str) -> None:
    """Print step-by-step portal instructions for adding Mail.Read."""
    print()
    print("Manual setup instructions:")
    print(
        "  1. Go to https://portal.azure.com"
        " > Microsoft Entra ID > App registrations"
    )
    print(f"  2. Find or create an app with client ID: {client_id}")
    print(
        "  3. Authentication > Add a platform > Mobile and desktop applications"
        " > check 'http://localhost' > Save"
    )
    print(
        "  4. API permissions > Add a permission"
        " > Microsoft Graph > Delegated permissions"
    )
    print("  5. Search for 'Mail.Read' and add it")
    print("  6. Click 'Grant admin consent' if required by your organization")
    print()
    print("  Alternatively, use --device-code to skip redirect URI setup.")


# ---------------------------------------------------------------------------
# Message download
# ---------------------------------------------------------------------------


async def download_messages(
    client: GraphServiceClient,
    output_dir: Path,
    max_results: int,
    filter_query: str,
    search_query: str,
) -> int:
    """Download messages from the signed-in user's mailbox as ``.eml`` files.

    Messages are saved as ``000001.eml``, ``000002.eml``, … in *output_dir*.
    Returns the number of messages successfully downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
        top=min(max_results, 100),
        select=["id", "subject", "receivedDateTime"],
        orderby=["receivedDateTime desc"],
    )
    if filter_query:
        query_params.filter = filter_query

    config = RequestConfiguration(query_parameters=query_params)

    # KQL $search requires the eventual-consistency header
    if search_query:
        query_params.search = f'"{search_query}"'
        config.headers.add("ConsistencyLevel", "eventual")

    count = 0
    response = await client.me.messages.get(request_configuration=config)

    while response and response.value:
        for msg in response.value:
            if count >= max_results:
                return count

            if not msg.id:
                continue
            # GET /me/messages/{id}/$value  → MIME content
            mime_content = await client.me.messages.by_message_id(msg.id).content.get()
            if mime_content is None:
                print(f"  [skip] empty MIME for message {msg.id}")
                continue

            eml_path = output_dir / f"{count + 1:06d}.eml"
            if isinstance(mime_content, bytes):
                eml_path.write_bytes(mime_content)
            else:
                # Some SDK versions return a stream-like object
                eml_path.write_bytes(mime_content.read())
            count += 1

            subject = msg.subject or "(no subject)"
            print(f"  [{count}] {subject}")

        # Follow @odata.nextLink for the next page of results
        if count < max_results and response.odata_next_link:
            response = await client.me.messages.with_url(response.odata_next_link).get()
        else:
            break

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point – routes to the requested action."""
    credential = get_credential(
        args.application_client_id,
        args.tenant_id,
        args.device_code,
        login_hint=args.client_id,
    )
    client = GraphServiceClient(credential, REQUIRED_SCOPES)

    if args.check_app_reg_permissions:
        ok = await check_permissions(client)
        await check_app_registration_permissions(credential, args.application_client_id)
        if not ok:
            return
        return

    if args.setup_permissions:
        await setup_permissions(credential, args.application_client_id)
        return

    # Default action: download messages (verify permissions first)
    if not await check_permissions(client):
        return

    print(f"Downloading up to {args.max_results} messages …")
    start_time = time.time()
    count = await download_messages(
        client, args.output_dir, args.max_results, args.filter, args.search
    )
    elapsed = time.time() - start_time
    print(f"Downloaded {count} messages to {args.output_dir} in {elapsed:.1f}s")


def main() -> None:
    """CLI entry point for Outlook mail dump."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download Outlook messages as .eml files via Microsoft Graph API",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=os.environ.get("OUTLOOK_CLIENT_ID"),
        help=("User email address for login / login_hint " "(env: OUTLOOK_CLIENT_ID)"),
    )
    parser.add_argument(
        "--application-client-id",
        type=str,
        default=os.environ.get("OUTLOOK_APPLICATION_CLIENT_ID"),
        help=(
            "Azure AD app registration client ID / GUID "
            "(env: OUTLOOK_APPLICATION_CLIENT_ID)"
        ),
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=os.environ.get("OUTLOOK_TENANT_ID", "common"),
        help="Azure AD tenant ID (env: OUTLOOK_TENANT_ID, default: 'common')",
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
        help="Output directory for .eml files (default: mail_dump)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help=(
            "OData $filter expression "
            "(e.g. \"from/emailAddress/address eq 'user@example.com'\")"
        ),
    )
    parser.add_argument(
        "--search",
        type=str,
        default="",
        help="KQL $search query (e.g. 'subject:quarterly report')",
    )
    parser.add_argument(
        "--device-code",
        action="store_true",
        help="Use device-code flow instead of interactive browser auth",
    )
    parser.add_argument(
        "--check-app-reg-permissions",
        action="store_true",
        help="Only verify that required Graph API permissions are available",
    )
    parser.add_argument(
        "--setup-permissions",
        action="store_true",
        help=(
            "Add required permissions to the app registration "
            "(requires Application.ReadWrite.All or Global Admin)"
        ),
    )
    args = parser.parse_args()

    # Validate --client-id format when provided
    if args.client_id:
        email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        if not email_pattern.match(args.client_id):
            parser.error(
                f"--client-id must be an email address (e.g. 'user@example.com'),"
                f" got: '{args.client_id}'"
            )

    # Validate --application-client-id format when provided
    if args.application_client_id:
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        if not uuid_pattern.match(args.application_client_id):
            parser.error(
                f"--application-client-id must be a GUID "
                f"(e.g. 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'),"
                f" got: '{args.application_client_id}'\n"
                "Find it in Azure Portal > Microsoft Entra ID > App registrations"
                " > your app > Application (client) ID"
            )

    if not args.application_client_id:
        print(
            "No --application-client-id or OUTLOOK_APPLICATION_CLIENT_ID set;"
            " falling back to DefaultAzureCredential."
        )

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
