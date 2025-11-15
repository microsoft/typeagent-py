# Run Tests

Run the TypeAgent test suite to verify functionality and ensure code quality.

## Usage

This skill allows you to:
- Run the complete test suite
- Run specific test files or test functions
- Run tests with coverage reporting
- Run type checking with Pyright
- Run code formatting checks

## How to use

### Run All Tests

```bash
cd /home/user/typeagent-py
make test
```

Or directly with pytest:
```bash
pytest
```

### Run Specific Test File

```bash
pytest test/test_conversation.py
```

### Run Specific Test Function

```bash
pytest test/test_conversation.py::test_create_conversation
```

### Run Tests with Pattern Matching

```bash
# Run all tests matching pattern
pytest -k "knowledge"

# Run all tests in a directory
pytest test/knowpro/
```

## Test Coverage

Run tests with coverage reporting:

```bash
make coverage
```

Or with pytest:
```bash
pytest --cov=typeagent --cov-report=html
```

View HTML coverage report:
```bash
open htmlcov/index.html
```

## Type Checking

Run Pyright type checker:

```bash
make check
```

Or directly:
```bash
pyright
```

## Code Formatting

Check code formatting:

```bash
make format
```

This runs Black formatter to ensure consistent code style.

## Test Organization

The test suite is organized by module:

```
test/
├── knowpro/           # Core knowledge processing tests
│   ├── test_conversation.py
│   ├── test_knowledge.py
│   ├── test_query.py
│   ├── test_search.py
│   └── ...
├── storage/           # Storage backend tests
│   ├── memory/
│   │   ├── test_collections.py
│   │   ├── test_indexes.py
│   │   └── ...
│   └── sqlite/
│       ├── test_collections.py
│       ├── test_indexes.py
│       └── ...
├── aitools/          # AI/ML utilities tests
│   ├── test_embeddings.py
│   └── ...
├── podcasts/         # Podcast processing tests
├── transcripts/      # Transcript processing tests
└── emails/           # Email import tests
```

## Running Specific Test Suites

### Knowledge Extraction Tests
```bash
pytest test/knowpro/test_knowledge.py
```

### Query Processing Tests
```bash
pytest test/knowpro/test_query.py
```

### Storage Backend Tests
```bash
# Memory backend
pytest test/storage/memory/

# SQLite backend
pytest test/storage/sqlite/
```

### Integration Tests
```bash
pytest test/integration/
```

## Test Options

### Verbose Output
```bash
pytest -v
```

### Show Print Statements
```bash
pytest -s
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

### Run Failed Tests First
```bash
pytest --ff
```

### Parallel Execution
```bash
pytest -n auto
```

## Common Test Scenarios

### Test Before Commit
```bash
# Run all checks
make format
make check
make test
```

### Quick Smoke Test
```bash
# Run fast tests only
pytest -m "not slow"
```

### Full Validation
```bash
# Format, type check, test, and coverage
make format
make check
make coverage
```

## Test Configuration

Configuration in `pytest.ini` or `pyproject.toml`:

```ini
[tool.pytest.ini_options]
testpaths = ["test"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

## Debugging Tests

### Run with Debugger
```bash
pytest --pdb
```

### Show Locals on Failure
```bash
pytest -l
```

### Increase Verbosity
```bash
pytest -vv
```

### Show Test Duration
```bash
pytest --durations=10
```

## Continuous Integration

The test suite runs in CI on:
- Every push to GitHub
- Every pull request
- Scheduled nightly builds

CI configuration in `.github/workflows/`.

## Writing New Tests

Test template:
```python
import pytest
from typeagent.knowpro import create_conversation

def test_my_feature():
    """Test description."""
    # Arrange
    conversation = create_conversation(name="test", backend="memory")

    # Act
    conversation.add_message({
        "text": "Test message",
        "sender": "User"
    })

    # Assert
    messages = conversation.get_messages()
    assert len(messages) == 1
    assert messages[0].text == "Test message"

@pytest.mark.slow
def test_slow_feature():
    """Test that takes a long time."""
    pass
```

## Test Best Practices

1. **Use fixtures** for common setup
2. **Mark slow tests** with `@pytest.mark.slow`
3. **Use memory backend** for faster tests
4. **Mock external APIs** to avoid rate limits
5. **Test both backends** (memory and SQLite)
6. **Clean up** test data after tests
7. **Use descriptive names** for test functions
