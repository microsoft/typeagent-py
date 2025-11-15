# Manage Storage Backends

Manage TypeAgent's dual storage system with Memory and SQLite backends.

## Usage

This skill allows you to:
- Choose between Memory (fast, temporary) and SQLite (persistent) backends
- Migrate data between backends
- Manage storage directories
- Configure storage settings
- Clear and reset storage

## How to use

### Choose Storage Backend

**Memory Backend** (fast, temporary):
```python
from typeagent.knowpro import create_conversation

conversation = create_conversation(
    name="temp-conv",
    backend="memory"
)
```

**SQLite Backend** (persistent):
```python
conversation = create_conversation(
    name="persistent-conv",
    backend="sqlite",
    data_dir="./data"
)
```

### Storage Backend Comparison

| Feature | Memory | SQLite |
|---------|--------|--------|
| Speed | Very Fast | Fast |
| Persistence | No | Yes |
| Data Size | Limited by RAM | Large datasets |
| Concurrent Access | Single process | Multi-process |
| Use Case | Testing, demos | Production |

## Manage Data Directory

### Set Custom Data Directory

```python
conversation = create_conversation(
    name="my-conv",
    backend="sqlite",
    data_dir="/path/to/my/data"
)
```

### Default Locations

- Default data directory: `./data`
- SQLite database: `{data_dir}/{conversation_name}.db`
- Embeddings cache: `{data_dir}/embeddings/`

### List Data Directories

```bash
cd /home/user/typeagent-py
ls -la ./data/
```

## Storage Operations

### Clear Storage

Remove all data for a conversation:

```bash
# Remove SQLite database
rm ./data/my-conversation.db

# Remove entire data directory
rm -rf ./data/
```

### Export/Backup Data

```bash
# Backup SQLite database
cp ./data/my-conversation.db ./backups/my-conversation-backup.db

# Backup entire data directory
tar -czf data-backup.tar.gz ./data/
```

### Restore Data

```bash
# Restore SQLite database
cp ./backups/my-conversation-backup.db ./data/my-conversation.db

# Restore entire data directory
tar -xzf data-backup.tar.gz
```

## Migration Between Backends

### Memory to SQLite

```python
from typeagent.knowpro import create_conversation

# Create in memory
mem_conv = create_conversation(name="temp", backend="memory")
mem_conv.add_message({"text": "Test message", "sender": "User"})

# Export messages
messages = mem_conv.get_messages()
refs = mem_conv.get_semantic_refs()

# Create in SQLite
sql_conv = create_conversation(name="persistent", backend="sqlite")

# Import data
for msg in messages:
    sql_conv.add_message({
        "text": msg.text,
        "sender": msg.sender,
        "timestamp": msg.timestamp
    })
```

### SQLite to Memory (for testing)

```python
# Load from SQLite
sql_conv = create_conversation(name="prod", backend="sqlite")
messages = sql_conv.get_messages()

# Create memory copy
mem_conv = create_conversation(name="test", backend="memory")
for msg in messages:
    mem_conv.add_message({
        "text": msg.text,
        "sender": msg.sender
    })
```

## Storage Configuration

### SQLite Settings

```python
from typeagent.storage.sqlite.provider import SQLiteStorageProvider

# Custom SQLite configuration
provider = SQLiteStorageProvider(
    db_path="./data/my-conv.db",
    enable_wal=True,        # Write-Ahead Logging
    cache_size=10000,       # Cache size
    synchronous="NORMAL"    # Sync mode
)
```

### Memory Settings

```python
from typeagent.storage.memory.provider import MemoryStorageProvider

# Memory provider (no special config needed)
provider = MemoryStorageProvider()
```

## Storage Provider API

Both backends implement the same interface:

```python
# Get storage provider
provider = conversation.storage_provider

# Collections
message_collection = provider.get_message_collection()
ref_collection = provider.get_semantic_ref_collection()

# Indexes
semantic_index = provider.get_semantic_ref_index()
property_index = provider.get_property_index()
message_index = provider.get_message_text_index()
timestamp_index = provider.get_timestamp_index()
related_terms = provider.get_related_terms_index()
threads = provider.get_conversation_threads()
```

## Monitoring Storage

### Check Database Size

```bash
# SQLite database size
ls -lh ./data/*.db

# Total data directory size
du -sh ./data/
```

### Count Records

```python
# Count messages
message_count = len(conversation.get_messages())
print(f"Messages: {message_count}")

# Count semantic references
ref_count = len(conversation.get_semantic_refs())
print(f"Knowledge items: {ref_count}")
```

### Database Statistics

For SQLite:
```bash
# Open database
sqlite3 ./data/my-conversation.db

# Show tables
.tables

# Count rows
SELECT COUNT(*) FROM messages;
SELECT COUNT(*) FROM semantic_refs;

# Database size
.dbinfo
```

## Best Practices

1. **Use Memory for:**
   - Unit tests
   - Quick experiments
   - Temporary processing
   - Small datasets

2. **Use SQLite for:**
   - Production deployments
   - Large datasets
   - Long-term storage
   - Multi-session usage

3. **Backup Regularly:**
   - Backup SQLite databases before major changes
   - Use version control for configurations
   - Test restore procedures

4. **Monitor Performance:**
   - Track database size growth
   - Monitor query performance
   - Optimize indexes if needed
