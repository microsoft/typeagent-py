# Run MCP Server

Run the Model Context Protocol (MCP) server to integrate TypeAgent with Claude and other AI clients.

## Usage

This skill allows you to:
- Expose TypeAgent functionality via MCP protocol
- Enable Claude to query the knowledge base
- Provide conversation management through MCP tools
- Integrate with Claude Desktop or other MCP clients

## How to use

When the user wants to run the MCP server:

1. Start the MCP server:
```bash
cd /home/user/typeagent-py
make mcp
```

Or run directly:
```bash
python -m typeagent.mcp.server
```

## MCP Tools Available

The server exposes these tools to MCP clients:

### 1. `query_knowledge`
Query the knowledge base.

Parameters:
- `query` (string) - The query to search for
- `thread` (string, optional) - Filter by conversation thread
- `max_results` (number, optional) - Maximum results to return

### 2. `add_message`
Add a message to the conversation.

Parameters:
- `text` (string) - Message text
- `sender` (string, optional) - Message sender
- `thread` (string, optional) - Conversation thread
- `timestamp` (string, optional) - ISO timestamp

### 3. `list_messages`
List all messages in the conversation.

Parameters:
- `thread` (string, optional) - Filter by thread
- `limit` (number, optional) - Maximum messages to return

### 4. `get_semantic_refs`
Get extracted knowledge references.

Parameters:
- `ref_type` (string, optional) - Filter by type (entity, topic, action, tag)
- `limit` (number, optional) - Maximum refs to return

## Claude Desktop Integration

To use with Claude Desktop:

1. Edit Claude Desktop config:
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add TypeAgent MCP server:
```json
{
  "mcpServers": {
    "typeagent": {
      "command": "python",
      "args": ["-m", "typeagent.mcp.server"],
      "cwd": "/home/user/typeagent-py",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

3. Restart Claude Desktop

4. TypeAgent tools will be available in Claude

## Configuration

Environment variables:
- `OPENAI_API_KEY` - OpenAI API key (required)
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint (optional)
- `AZURE_OPENAI_API_KEY` - Azure OpenAI key (optional)
- `TYPEAGENT_DATA_DIR` - Data directory (default: ./data)
- `TYPEAGENT_BACKEND` - Backend type (default: sqlite)

## Example Usage in Claude

Once configured, you can ask Claude:

```
"Query my TypeAgent knowledge base for information about AI topics"
```

Claude will use the `query_knowledge` tool to search your knowledge base.

```
"Add this message to my knowledge base: The meeting is scheduled for tomorrow at 3pm"
```

Claude will use the `add_message` tool to add the message.

## Development Mode

Run with debug logging:
```bash
python -m typeagent.mcp.server --debug
```

Test the server:
```bash
# In another terminal
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python -m typeagent.mcp.server
```

## Implementation

The MCP server:
- Implements MCP protocol specification
- Provides stdio-based communication
- Exposes TypeAgent functionality as MCP tools
- Handles conversation state management
- Supports concurrent requests
- Provides error handling and logging
