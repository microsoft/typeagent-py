# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This is Guido's Makefile. Please don't make it complicated.

.PHONY: all
all: venv format check test build

.PHONY: format
format: venv
	uv run isort src tests tools examples $(FLAGS)
	uv run black -tpy312 src tests tools examples $(FLAGS)

.PHONY: check
check: venv
	uv run pyright src tests tools examples

.PHONY: test
test: venv
	uv run pytest $(FLAGS)

.PHONY: coverage
coverage: venv
	coverage erase
	COVERAGE_PROCESS_START=.coveragerc uv run coverage run -m pytest $(FLAGS)
	coverage combine
	coverage report

.PHONY: demo
demo: venv
	uv run python -m tools.query $(FLAGS)

.PHONY: compare
compare: venv
	uv run python -m tools.query --batch $(FLAGS)

.PHONY: eval
eval: venv
	rm -f eval.db
	uv run python tools/load_json.py --database eval.db tests/testdata/Episode_53_AdrianTchaikovsky_index
	uv run python tools/query.py --batch --database eval.db --answer-results tests/testdata/Episode_53_Answer_results.json --search-results tests/testdata/Episode_53_Search_results.json $(FLAGS)

.PHONY: mcp
mcp: venv
	uv run mcp dev src/typeagent/mcp/server.py

.PHONY: profile
profile: venv
	</dev/null uv run python -m cProfile -s ncalls -m test.cmpsearch --interactive --podcast ~/AISystems-Archive/data/knowpro/test/indexes/All_Episodes_index | head -60

.PHONY: scaling
scaling: venv
	</dev/null uv run python -m test.cmpsearch --interactive --podcast ~/AISystems-Archive/data/knowpro/test/indexes/All_Episodes_index

.PHONY: build
build: venv
	uv build

.PHONY: release
release: venv
	uv run python tools/release.py $(VERSION)

.PHONY: venv
venv: .venv

.venv:
	@echo "(If 'uv' fails with 'No such file or directory', try 'make install-uv')"
	uv sync -q $(FLAGS)
	uv run black --version
	@echo "(If 'pyright' fails with 'error while loading shared libraries: libatomic.so.1:', try 'make install-libatomic')"
	uv run pyright --version
	uv run pytest --version

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
