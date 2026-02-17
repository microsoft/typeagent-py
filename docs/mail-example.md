# Mail Example — End-to-End Workflow

Working with emails in TypeAgent follows three steps:
**download → ingest → query**.

## Step 1: Download emails

Fetch raw `.eml` files from your mail source into a local folder.
Choose whichever provider applies to you:

| Source | Tool | Details |
|---|---|---|
| **Gmail** | `tools/mail/gmail_dump.py` | Uses the Gmail API with OAuth 2.0 |
| **Outlook** | `tools/mail/outlook_dump.py` | Uses the Microsoft Graph API |
| **Mbox file** | `tools/mail/mbox_dump.py` | Extracts from a local or remote `.mbox` archive |

See [`tools/mail/README.md`](../tools/mail/README.md) for provider-specific
setup and usage instructions.

After this step you will have a folder (e.g. `mail_dump/`) containing
individual `.eml` files.

## Step 2: Ingest emails

Run the ingestion tool to parse the `.eml` files, extract structure and
embeddings, and store everything in a local database:

```bash
python tools/ingest_email.py <path-to-eml-folder>
```

This creates (or updates) a database that the query tool can search against.

## Step 3: Query

Start an interactive query session against the ingested email database:

```bash
python tools/query.py <database-path>
```

You can ask natural-language questions about your emails (senders, topics,
dates, content, etc.) and the system will use Structured RAG to retrieve
relevant results.