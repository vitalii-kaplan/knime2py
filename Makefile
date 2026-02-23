SHELL := /bin/bash

.DEFAULT_GOAL := help

PYTHON ?= python3
VERSION ?=
MESSAGE ?=
MODE ?= online
Q ?=
TARGET_DIR ?= src/
WORKFLOW ?= tests/data/KNIME_single_csv
OUT ?= output

.PHONY: help
help: ## Show available targets with descriptions.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"

.PHONY: venv
venv: ## Create local .venv and print activation command for current shell.
	@python3 -m venv .venv
	@printf "Virtualenv created at .venv\n"
	@printf "Activate in current shell with:\n  source .venv/bin/activate\n"

.PHONY: venv-shell
venv-shell: ## Open a new shell session with .venv activated.
	@python3 -m venv .venv
	@bash -lc 'source .venv/bin/activate && exec "$$SHELL" -i'

.PHONY: install
install: ## Install package + extras via scripts/install.sh (dev, rag, ml-examples by default).
	@bash scripts/install.sh

.PHONY: install-k2p
install-k2p: ## Create .venv, install editable package, run k2p smoke checks.
	@bash scripts/install_k2p.sh

.PHONY: test
test: ## Run pytest suite.
	@pytest

.PHONY: test-cov
test-cov: ## Run coverage check workflow (scripts/test_cov_check.sh).
	@bash scripts/test_cov_check.sh

.PHONY: test-gen
test-gen: ## Run KNIME test fixture cleanup/generator wrapper.
	@bash scripts/test_gen.sh

.PHONY: k2p
k2p: ## Run knime2py once (python -m knime2py $(WORKFLOW) --out $(OUT)).
	@$(PYTHON) -m knime2py "$(WORKFLOW)" --out "$(OUT)"

.PHONY: k2p-script
k2p-script: ## Run opinionated local wrapper from scripts/k2p.sh.
	@bash scripts/k2p.sh

.PHONY: docker-run
docker-run: ## Run GHCR image wrapper (scripts/k2p_docker.sh).
	@bash scripts/k2p_docker.sh

.PHONY: docker-dev
docker-dev: ## Build + run local dev Docker image (scripts/docker_local_build_run.sh).
	@bash scripts/docker_local_build_run.sh

.PHONY: docker-pull
docker-pull: ## Pull published GHCR image and show help.
	@bash scripts/docker_pull.sh

.PHONY: pex-build
pex-build: ## Build local PEX via scripts/pex_local_build.sh MODE=$(MODE).
	@bash scripts/pex_local_build.sh "$(MODE)"

.PHONY: pex-run
pex-run: ## Run locally built PEX wrapper from scripts/k2p_pex.sh.
	@bash scripts/k2p_pex.sh

.PHONY: docs
docs: ## Install docs deps and serve MkDocs (scripts/doc_generator.sh).
	@bash scripts/doc_generator.sh

.PHONY: docs-build
docs-build: ## Build MkDocs site locally (CI-equivalent for gh-pages build step).
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install mkdocs mkdocs-material "mkdocstrings[python]" mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index griffe
	@mkdocs build

.PHONY: rag-index
rag-index: ## Rebuild local RAG index (scripts/rag_index_rebuild.sh).
	@bash scripts/rag_index_rebuild.sh

.PHONY: rag-query
rag-query: ## Query RAG via OpenAI; pass Q="your question".
	@if [ -z "$(Q)" ]; then echo 'Usage: make rag-query Q="your question"'; exit 2; fi
	@$(PYTHON) -m rag.query_openai "$(Q)"

.PHONY: rag-edit-batch
rag-edit-batch: ## Batch rewrite Python files with RAG prompt; TARGET_DIR=src/ by default.
	@bash scripts/rag_query_files.sh "$(TARGET_DIR)"

.PHONY: ollama-start
ollama-start: ## Install/start Ollama and pull llama3 (scripts/ollama_start.sh).
	@bash scripts/ollama_start.sh

.PHONY: release
release: ## Run release helper; requires VERSION and optional MESSAGE.
	@if [ -z "$(VERSION)" ]; then echo 'Usage: make release VERSION=x.y.z [MESSAGE="note"]'; exit 2; fi
	@bash scripts/release.sh "$(VERSION)" "$(MESSAGE)"

.PHONY: git-history
git-history: ## Show commit history since latest semver tag (scripts/git_history.sh).
	@bash scripts/git_history.sh

.PHONY: clean
clean: ## Remove common generated artifacts.
	@rm -rf output tests/data/\!output .rag_index

.PHONY: ci-tests
ci-tests: ## Run local equivalent of .github/workflows/tests.yml (coverage included).
	@$(PYTHON) -m pip install -U "pip>=24,<26"
	@$(PYTHON) -m pip install -e ".[dev,ml-examples]"
	@pytest --cov=knime2py --cov-report=xml:coverage.xml

.PHONY: ci-wheel
ci-wheel: ## Build wheel locally (used by GitHub build workflows).
	@$(PYTHON) -m pip install -U "pip>=24,<26"
	@$(PYTHON) -m pip install "build==1.2.1"
	@$(PYTHON) -m build --wheel

.PHONY: ci-pex
ci-pex: ## Build PEX for current OS/arch locally (GitHub does matrix across OSes).
	@$(PYTHON) -m pip install -U "pip>=24,<26"
	@$(PYTHON) -m pip install "build==1.2.1" "pex==2.74.1"
	@bash scripts/pex_local_build.sh online

.PHONY: ci-docker
ci-docker: ## Build Docker image locally (release-docker build equivalent, no push).
	@docker build --pull --no-cache -t knime2py:local-ci .

.PHONY: ci-local
ci-local: ci-tests ci-wheel ci-pex docs-build ci-docker ci-note ## Run local pre-push CI/build pipeline.

.PHONY: ci-note
ci-note: ## Show what GitHub-only matrix/release jobs are not fully reproduced locally.
	@printf "Note: GitHub matrix-only jobs are not fully local:\n"
	@printf "  - build-exe.yml (Windows PyInstaller/Nuitka artifacts)\n"
	@printf "  - build-pex.yml cross-OS matrix (this runs current OS only)\n"
	@printf "  - release-* publish/push steps and artifact uploads\n"
