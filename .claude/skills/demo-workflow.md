# Demo Workflow

Run complete demonstration workflows showing TypeAgent's capabilities.

## Usage

This skill demonstrates:
- End-to-end ingestion and query workflow
- Real-world usage examples
- Best practices for using TypeAgent
- Common patterns and use cases

## How to use

### Interactive Demo

Run the interactive query demo:

```bash
cd /home/user/typeagent-py
make demo
```

This starts an interactive session where you can:
- Query the knowledge base
- See real-time search results
- View citations and sources
- Explore different query types

### Complete Workflow Demo

Full demonstration from ingestion to querying:

```bash
cd /home/user/typeagent-py
python demo/ingest.py
python demo/query.py
```

## Demo Scenarios

### 1. Podcast Ingestion and Query

**Step 1: Ingest a podcast**
```bash
# Create sample podcast file
cat > /tmp/sample-podcast.txt << 'EOF'
Host: Welcome to the AI Tech Podcast! Today we're discussing the future of artificial intelligence with Dr. Sarah Chen, a leading researcher in machine learning.

Dr. Chen: Thanks for having me! I'm excited to talk about recent developments in neural networks and deep learning.

Host: Let's start with transformers. How have they changed the field?

Dr. Chen: Transformers have revolutionized natural language processing. The attention mechanism allows models to understand context much better than previous architectures like RNNs or LSTMs.

Host: What about GPT-4 and other large language models?

Dr. Chen: Large language models represent a paradigm shift. They demonstrate emergent capabilities that weren't explicitly programmed. However, we need to be careful about hallucinations and bias.
EOF

# Ingest the podcast
python -m tools.ingest_podcast /tmp/sample-podcast.txt --thread "ai-podcast-ep1"
```

**Step 2: Query the knowledge base**
```bash
# Interactive queries
python -m tools.query --interactive

# Try these queries:
# - "What did Dr. Chen say about transformers?"
# - "What are the concerns with large language models?"
# - "Who are the speakers in this podcast?"
```

### 2. Transcript Ingestion Demo

**Step 1: Create a VTT transcript**
```bash
cat > /tmp/meeting.vtt << 'EOF'
WEBVTT

00:00:01.000 --> 00:00:05.000
Welcome everyone to the Q1 planning meeting.

00:00:06.000 --> 00:00:12.000
Today we'll discuss our product roadmap and key initiatives.

00:00:13.000 --> 00:00:20.000
Alice: I think we should prioritize the mobile app redesign.

00:00:21.000 --> 00:00:28.000
Bob: Agreed. We also need to focus on performance improvements.

00:00:29.000 --> 00:00:35.000
Alice: The analytics dashboard is critical for our enterprise customers.
EOF

# Ingest transcript
python -m tools.ingest_vtt /tmp/meeting.vtt --thread "q1-planning"
```

**Step 2: Query with temporal context**
```bash
python -m tools.query --thread "q1-planning" --query "What priorities were discussed?"
```

### 3. Email Import Demo

**Step 1: Import emails** (requires Gmail setup)
```bash
# Import recent work emails
python -m typeagent.emails.email_import \
  --query "from:colleague@company.com" \
  --max-results 20 \
  --thread "work-emails"
```

**Step 2: Query email content**
```bash
python -m tools.query \
  --thread "work-emails" \
  --query "What action items were mentioned in emails?"
```

### 4. Multi-Source Knowledge Base

**Build a comprehensive knowledge base:**

```bash
# Ingest podcast
python -m tools.ingest_podcast ./data/podcast1.txt --thread "podcasts"

# Ingest meeting transcript
python -m tools.ingest_vtt ./data/meeting.vtt --thread "meetings"

# Ingest emails
python -m typeagent.emails.email_import --thread "emails" --max-results 50

# Query across all sources
python -m tools.query --query "What has been discussed about AI this month?"
```

### 5. Knowledge Extraction Demo

**See what knowledge is extracted:**

```python
from typeagent.knowpro import create_conversation

# Create conversation with extraction enabled
conv = create_conversation(
    name="demo",
    backend="memory",
    settings={"enable_knowledge_extraction": True}
)

# Add a rich message
conv.add_message({
    "text": """
    Yesterday, the research team at Stanford University published a paper
    on quantum computing. Dr. Emily Rodriguez led the project, which
    focused on error correction in quantum circuits. The team demonstrated
    a 50% improvement in qubit stability using novel error mitigation
    techniques. This breakthrough could accelerate practical quantum
    applications in cryptography and drug discovery.
    """,
    "sender": "Science News Bot"
})

# View extracted knowledge
refs = conv.get_semantic_refs()

print("\n=== Extracted Knowledge ===")
for ref in refs:
    print(f"\nType: {ref.type}")
    print(f"Value: {ref.value}")
    if ref.description:
        print(f"Description: {ref.description}")
    print(f"Properties: {ref.properties}")
```

Expected output:
- **Entities**: Stanford University, Dr. Emily Rodriguez
- **Topics**: quantum computing, error correction, cryptography, drug discovery
- **Actions**: published (paper), led (project), demonstrated (improvement)
- **Tags**: research, breakthrough, quantum circuits

### 6. Search Comparison Demo

**Compare different search strategies:**

```python
from typeagent.knowpro import create_conversation

conv = create_conversation(name="search-demo", backend="memory")

# Add sample messages
messages = [
    "Python is a popular programming language for AI development.",
    "Machine learning models require large datasets for training.",
    "TensorFlow and PyTorch are widely used ML frameworks."
]

for msg in messages:
    conv.add_message({"text": msg, "sender": "Bot"})

# 1. Keyword search (via semantic ref index)
print("=== Keyword Search: 'Python' ===")
refs = conv.storage_provider.get_semantic_ref_index().search("Python")
for ref in refs:
    print(f"  {ref.value}")

# 2. Semantic search (via message text index)
print("\n=== Semantic Search: 'AI programming tools' ===")
msgs = conv.storage_provider.get_message_text_index().search(
    "AI programming tools", limit=3
)
for msg in msgs:
    print(f"  {msg.text[:50]}... (score: {msg.score:.3f})")

# 3. Full query (combines all indexes)
print("\n=== Full Query: 'What frameworks are used for ML?' ===")
result = conv.query("What frameworks are used for ML?")
print(f"Answer: {result.answer}")
```

## Performance Benchmarks

**Run benchmark tests:**

```bash
# Benchmark ingestion speed
time python -m tools.ingest_podcast large-podcast.txt

# Benchmark query speed
time python -m tools.query --query "test query"

# Compare backends
time python -m tools.query --backend memory --query "test"
time python -m tools.query --backend sqlite --query "test"
```

## Demo Data

Sample data is available in:
- `demo/` - Demo scripts
- `examples/` - Example code
- `test/fixtures/` - Test data files

## Interactive Python Session

Explore TypeAgent interactively:

```python
from typeagent.knowpro import create_conversation

# Create conversation
conv = create_conversation(name="interactive", backend="memory")

# Add messages
conv.add_message({"text": "Hello world!", "sender": "User"})

# Query
result = conv.query("What did the user say?")
print(result.answer)

# Explore
messages = conv.get_messages()
refs = conv.get_semantic_refs()
```

## Best Practices Demonstrated

1. **Enable knowledge extraction** for rich queries
2. **Use threads** to organize conversations
3. **Choose appropriate backend** (memory for testing, SQLite for production)
4. **Query with context** for better results
5. **Batch processing** for large datasets
6. **Monitor performance** and optimize as needed
