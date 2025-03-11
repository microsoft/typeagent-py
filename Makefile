# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This is Guido's Makefile. Please don't make it complicated.

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "make help   # Help"
	@echo "make all    # test, demo, build"
	@echo "make test   # Run pyright"
	@echo "make demo   # Run import_podcast demo"
	@echo "make build  # Build the wheel (under dist/)"
	@echo "make venv   # Create venv/"
	@echo "make clean  # Remove build/, dist/, venv/"

.PHONY: all
all: test demo build

.PHONY: test
# To install pyright, use `npm install -g pyright` .
test:
	pyright typeagent

.PHONY: build
build: venv
	venv/bin/python -m build --wheel

.PHONY: demo
demo: venv
	venv/bin/python -m typeagent.memconv testdata/npr.txt

# Not phony -- the venv directory is the product of this rule.
venv:
	python -m venv venv || (rm -rf venv && exit 1)
	venv/bin/pip install build

.PHONY: clean
clean:
	rm -rf build dist venv *.egg-info
