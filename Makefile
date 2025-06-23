# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This is Guido's Makefile. Please don't make it complicated.

.PHONY: all
all: venv format check test build

.PHONY: format
format: venv
	.venv/bin/black typeagent test

.PHONY: check
check: venv
	.venv/bin/pyright --pythonpath .venv/bin/python typeagent test

.PHONY: test
test: venv
	.venv/bin/python -m pytest test

.PHONY: demo
demo: venv
	.venv/bin/python -m typeagent.demo

.PHONY: eval
eval: venv
	.venv/bin/python -m test.cmpsearch

.PHONY: build
build: venv
	.venv/bin/python -m build --wheel

.PHONY: venv
venv: .venv

.venv:
	@echo "(If 'uv' fails with 'No such file or directory', try 'make install-uv')"
	uv sync -q
	@.venv/bin/black --version | sed 's/, / /'
	@.venv/bin/pyright --version
	@.venv/bin/pytest --version

install-uv:
	curl -Ls https://astral.sh/uv/install.sh | sh

.PHONY: clean
clean:
	rm -rf build dist venv .venv *.egg-info
	rm -f *_data.json *_embedding.bin
	find . -type d -name __pycache__ | xargs rm -rf

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "make help   # Help (this message)"
	@echo "make        # Same as 'make all'"
	@echo "make all    # venv, format, check, test, build"
	@echo "make format # Run black"
	@echo "make check  # Run pyright"
	@echo "make test   # Run pytest (tests are in test/)"
	@echo "make build  # Build the wheel (under dist/)"
	@echo "make venv   # Create .venv/"
	@echo "make clean  # Remove build/, dist/, .venv/, *.egg-info/"
	@echo "make install-uv  # Install uv (if not already installed)"
