# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This is Guido's Makefile. Please don't make it complicated.

.PHONY: all
all: venv format check test build

.PHONY: format
format: venv
	.venv/bin/isort src tests tools examples $(FLAGS)
	.venv/bin/black -tpy312 -tpy313 -tpy314 src tests tools examples $(FLAGS)

.PHONY: check
check: venv
	.venv/bin/pyright --pythonpath .venv/bin/python src tests tools examples

.PHONY: test
test: venv
	.venv/bin/pytest $(FLAGS)

.PHONY: coverage
coverage: venv
	coverage erase
	COVERAGE_PROCESS_START=.coveragerc .venv/bin/coverage run -m pytest $(FLAGS)
	coverage combine
	coverage report

.PHONY: demo
demo: venv
	.venv/bin/python -m tools.query $(FLAGS)

.PHONY: compare
compare: venv
	.venv/bin/python -m tools.query --batch $(FLAGS)

.PHONY: mcp
mcp: venv
	.venv/bin/mcp dev src/typeagent/mcp/server.py

.PHONY: profile
profile: venv
	</dev/null .venv/bin/python -m cProfile -s ncalls -m test.cmpsearch --interactive --podcast ~/AISystems-Archive/data/knowpro/test/indexes/All_Episodes_index | head -60

.PHONY: scaling
scaling: venv
	</dev/null .venv/bin/python -m test.cmpsearch --interactive --podcast ~/AISystems-Archive/data/knowpro/test/indexes/All_Episodes_index

.PHONY: build
build: venv
	uv build

.PHONY: release
release: venv
	.venv/bin/python tools/release.py $(VERSION)

.PHONY: venv
venv: .venv

.venv:
	@echo "(If 'uv' fails with 'No such file or directory', try 'make install-uv')"
	uv sync -q $(FLAGS)
	.venv/bin/black --version
	@echo "(If 'pyright' fails with 'error while loading shared libraries: libatomic.so.1:', try 'make install-libatomic')"
	.venv/bin/pyright --version
	.venv/bin/pytest --version

.PHONY: sync
sync:
	uv sync $(FLAGS)

.PHONY: install-uv
install-uv:
	curl -Ls https://astral.sh/uv/install.sh | sh

.PHONY: install-libatomic
install-libatomic:
	sudo apt-get update
	sudo apt-get install -y libatomic1

.PHONY: clean
clean:
	rm -rf build dist venv .venv *.egg-info
	rm -f *_data.json *_embedding.bin
	find . -type d -name __pycache__ | xargs rm -rf

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "make help        # Help (this message)"
	@echo "make             # Same as 'make all'"
	@echo "make all         # venv, format, check, test, build"
	@echo "make format      # Run isort and black"
	@echo "make check       # Run pyright"
	@echo "make test        # Run pytest (tests are in tests/)"
	@echo "make coverage    # Run tests with coverage"
	@echo "make build       # Build the wheel (under dist/)"
	@echo "make demo        # python tools/query.py (interactive)"
	@echo "make compare     # python tools/query.py --batch"
	@echo "make venv        # Create .venv/"
	@echo "make sync        # Sync dependencies with uv"
	@echo "make clean       # Remove build/, dist/, .venv/, *.egg-info/"
	@echo "make install-uv  # Install uv (if not already installed)"
	@echo "make install-libatomic  # Install libatomic (if not already installed)"
