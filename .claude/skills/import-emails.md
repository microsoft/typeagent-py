# Import Gmail Emails

Import and process Gmail emails into the TypeAgent knowledge base with knowledge extraction.

## Usage

This skill allows you to:
- Import emails from Gmail using the Gmail API
- Extract knowledge from email content
- Track email metadata (sender, recipients, timestamps)
- Build searchable indexes for email content

## How to use

When the user wants to import emails:

1. Set up Gmail API credentials (required first time):
   - Enable Gmail API in Google Cloud Console
   - Download OAuth credentials JSON
   - Save as `credentials.json` in the project directory

2. Run the email import:
```bash
cd /home/user/typeagent-py
python -m typeagent.emails.email_import [options]
```

## Options

- `--data-dir PATH` - Directory to store the knowledge base (default: ./data)
- `--backend {memory,sqlite}` - Storage backend (default: sqlite)
- `--max-results N` - Maximum number of emails to import (default: 100)
- `--query "QUERY"` - Gmail search query (default: all emails)
- `--extract-knowledge` - Enable knowledge extraction (default: true)
- `--thread NAME` - Conversation thread name for emails

## Examples

Import recent emails:
```bash
python -m typeagent.emails.email_import --max-results 50
```

Import emails matching search:
```bash
python -m typeagent.emails.email_import --query "from:someone@example.com" --max-results 20
```

Import to specific thread:
```bash
python -m typeagent.emails.email_import --thread "work-emails-2024"
```

## Gmail Search Query Syntax

You can use Gmail's search syntax:
- `from:user@example.com` - Emails from specific sender
- `to:user@example.com` - Emails to specific recipient
- `subject:meeting` - Emails with subject containing "meeting"
- `after:2024/01/01` - Emails after date
- `has:attachment` - Emails with attachments
- `is:unread` - Unread emails

## First-Time Setup

1. Go to https://console.cloud.google.com
2. Create a new project or select existing
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials JSON
6. Save as `credentials.json` in project directory
7. Run import - browser will open for authorization

## Implementation

The email import pipeline:
1. Authenticate with Gmail API using OAuth
2. Fetch emails matching query
3. Parse email content and metadata
4. Convert to universal message format
5. Extract knowledge (entities, topics, actions)
6. Build searchable indexes
7. Store in chosen backend
