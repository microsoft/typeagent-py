# Mail Dump Tools

## Gmail (`gmail_dump.py`)

Downloads Gmail messages as `.eml` files via the Gmail API.

### Prerequisites

- Python 3.12+
- A Google Cloud project with the Gmail API enabled
- OAuth 2.0 client credentials (`client_secret.json`)

### Google Cloud Project Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Go to **APIs & Services** → **Library**
4. Search for **Gmail API** and click **Enable**

#### Create OAuth 2.0 credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. If prompted, configure the **OAuth consent screen** first:
   - Choose **External** (or **Internal** for organization-only)
   - Fill in the required fields (app name, email)
   - Add the scope `https://www.googleapis.com/auth/gmail.readonly`
   - Add your email as a test user (required for External apps in testing mode)
4. Back in **Credentials**, select **Desktop app** as the application type
5. Click **Create** and download the JSON file
6. Save it as `client_secret.json` in your credentials directory

### Configuration

Place `client_secret.json` in the directory you pass via `--creds-dir`
(defaults to the current directory). On first run the tool opens a browser
for OAuth consent and saves the resulting token as `token.json` in the same
directory. Subsequent runs reuse the cached token.

### Usage

```bash
# Download 50 most recent messages
python tools/mail/gmail_dump.py

# Download 200 messages
python tools/mail/gmail_dump.py --max-results 200

# Filter messages with a Gmail search query
python tools/mail/gmail_dump.py --query "from:alice@example.com"

# Specify a custom credentials directory
python tools/mail/gmail_dump.py --creds-dir ~/gmail-creds

# Specify a custom output directory
python tools/mail/gmail_dump.py --output-dir ~/my-emails
```

### Command-line flags

| Flag | Description | Default |
|---|---|---|
| `--max-results` | Max messages to download | `50` |
| `--output-dir` | Output directory for `.eml` files | `mail_dump` |
| `--query` | Gmail search query (same syntax as the Gmail search bar) | _(all messages)_ |
| `--creds-dir` | Directory containing `client_secret.json` and `token.json` | `.` (current dir) |

> **Note:** Messages are saved as `{message_id}.eml` (using the Gmail message
> ID as filename), unlike the Outlook tool which uses sequential numbering.

## Outlook (`outlook_dump.py`)

Downloads Outlook emails as `.eml` files via the Microsoft Graph API.

### Prerequisites

- Python 3.12+
- An Azure AD (Microsoft Entra ID) app registration

### Azure AD App Registration Setup

1. Go to [portal.azure.com](https://portal.azure.com) → **Microsoft Entra ID** → **App registrations**
2. Click **New registration** (or open an existing app)
3. Set a name (e.g. `outlook-mail-dump`) and click **Register**
4. Note the **Application (client) ID** (a GUID) — this is your `--application-client-id`
5. Note the **Directory (tenant) ID** — this is your `--tenant-id`

#### Add a redirect URI (required for interactive browser auth)

1. In your app registration, go to **Authentication**
2. Click **Add a platform** → **Mobile and desktop applications**
3. Check **`http://localhost`**
4. Click **Configure** / **Save**

> **Tip:** If you cannot add a redirect URI (e.g. on a headless server), use
> `--device-code` instead — it does not require a redirect URI.

#### Add API permissions

1. In your app registration, go to **API permissions**
2. Click **Add a permission** → **Microsoft Graph** → **Delegated permissions**
3. Search for **`Mail.Read`** and add it
4. Click **Grant admin consent** if required by your organization

### Configuration

The tool reads defaults from a `.env` file in the project root.
Add these variables:

```env
# User email address for login (pre-fills the sign-in page)
OUTLOOK_CLIENT_ID=user@example.onmicrosoft.com

# Azure AD app registration client ID (GUID)
OUTLOOK_APPLICATION_CLIENT_ID=...

# Azure AD tenant ID (GUID)
OUTLOOK_TENANT_ID=...
```

All values can also be overridden via command-line flags.

> **All three identity parameters are optional.** When `--application-client-id`
> (or `OUTLOOK_APPLICATION_CLIENT_ID`) is not provided, the tool falls back to
> `DefaultAzureCredential`, which automatically picks up ambient credentials in
> this order:
>
> 1. **Azure CLI** — run `az login` first
> 2. **VS Code** — sign in via the Azure Account extension
> 3. **Managed Identity** — on Azure VMs / App Service
> 4. **Environment variables** — `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`
>
> When `--client-id` is omitted the browser sign-in page will not be
> pre-filled (login hint). When `--tenant-id` is omitted it defaults to
> `"common"` (multi-tenant).
>
> This means you can run `python tools/mail/outlook_dump.py` with **zero
> configuration** as long as you are signed into Azure CLI or VS Code.

### Usage

```bash
# Download 50 most recent messages (interactive browser auth)
python tools/mail/outlook_dump.py

# Use device-code flow (no redirect URI needed)
python tools/mail/outlook_dump.py --device-code

# Download 200 messages
python tools/mail/outlook_dump.py --max-results 200

# Filter by sender
python tools/mail/outlook_dump.py \
    --filter "from/emailAddress/address eq 'alice@example.com'"

# Full-text search (KQL)
python tools/mail/outlook_dump.py --search "subject:quarterly report"

# Check permissions only
python tools/mail/outlook_dump.py --check-permissions

# Add Mail.Read to the app registration (requires admin privileges)
python tools/mail/outlook_dump.py --setup-permissions
```

### Command-line flags

| Flag | Description | Env variable |
|---|---|---|
| `--client-id` | User email for login hint | `OUTLOOK_CLIENT_ID` |
| `--application-client-id` | Azure AD app GUID | `OUTLOOK_APPLICATION_CLIENT_ID` |
| `--tenant-id` | Azure AD tenant ID | `OUTLOOK_TENANT_ID` |
| `--max-results` | Max messages to download (default: 50) | — |
| `--output-dir` | Output directory (default: `mail_dump`) | — |
| `--filter` | OData `$filter` expression | — |
| `--search` | KQL `$search` query | — |
| `--device-code` | Use device-code auth flow | — |
| `--check-permissions` | Verify Mail.Read access only | — |
| `--setup-permissions` | Add Mail.Read to app registration | — |

## Mbox (`mbox_dump.py`)

Extracts emails from a local or remote `.mbox` file into individual `.eml`
files (numbered `1.eml`, `2.eml`, …).

### Prerequisites

- Python 3.12+
- An `.mbox` file (local or accessible via URL)

No API keys, OAuth credentials, or cloud projects are required.

### How to obtain an mbox file

| Source | How |
|---|---|
| **Gmail** | Google Takeout → select **Mail** → export as `.mbox` |
| **Thunderbird** | ImportExportTools NG add-on → right-click folder → Export as mbox |
| **Apple Mail** | Mailbox → Export Mailbox… |
| **Mailing list archives** | Many lists (e.g. Mailman) offer `.mbox` downloads |

### Usage

```bash
# Extract emails from a local mbox file
# Creates a folder named "mailbox/" alongside the file
python tools/mail/mbox_dump.py mailbox.mbox

# Download an mbox from a URL, then extract
python tools/mail/mbox_dump.py --url https://example.com/archive.mbox

# Download and save with a custom local filename
python tools/mail/mbox_dump.py --url https://example.com/archive.mbox -o local.mbox

# Download from URL and also specify the mbox path to extract
python tools/mail/mbox_dump.py archive.mbox --url https://example.com/archive.mbox
```

### Command-line flags

| Flag | Description | Default |
|---|---|---|
| `mbox` (positional) | Path to the local `.mbox` file to extract | _(required unless `--url` is used)_ |
| `--url` | URL to download an `.mbox` file from | — |
| `-o` / `--output` | Local filename for the downloaded mbox (used with `--url`) | derived from URL |

> **Note:** The output directory is created automatically with the same name
> as the mbox file (without the extension). For example, `archive.mbox`
> produces an `archive/` folder containing `1.eml`, `2.eml`, etc.
