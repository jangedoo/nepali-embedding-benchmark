SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= uv run
SITE_DIR := site
BASE_PATH ?= /
EXPORT_DIR ?= $(SITE_DIR)/public/data/v1

.PHONY: help sync validate test test-contracts lint format check export queue publish \
	publish-verified publish-community package site-install site-test site-build site-check \
	site-dev audit clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "; printf "NEB development targets:\n\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install locked Python development dependencies
	uv sync --locked --extra dev

validate: ## Validate registries and canonical result runs
	$(PYTHON) neb validate

test: ## Run fast Python tests
	$(PYTHON) pytest -m 'not contract'

test-contracts: ## Run pinned Hugging Face dataset contract tests
	NEB_CONTRACT_TESTS=1 $(PYTHON) pytest tests/test_contract_datasets.py

lint: ## Check Python lint and formatting
	$(PYTHON) ruff check src tests
	$(PYTHON) ruff format --check src tests

format: ## Apply Python lint fixes and formatting
	$(PYTHON) ruff check --fix src tests
	$(PYTHON) ruff format src tests

check: lint test validate ## Run the fast Python verification suite

export: ## Regenerate versioned dashboard JSON and CSV artifacts
	$(PYTHON) neb export --output $(EXPORT_DIR)

queue: ## List model/task pairs missing verified results
	$(PYTHON) neb queue

publish: ## Publish all task runs for MODEL with STATUS, then refresh exports
	@if [[ ! "$(MODEL)" =~ ^[a-z0-9][a-z0-9._-]*$$ ]]; then \
		echo "Usage: make publish-verified MODEL=<registry-model-id>"; \
		exit 2; \
	fi
	@if [[ "$(STATUS)" != "verified" && "$(STATUS)" != "community" ]]; then \
		echo "STATUS must be 'verified' or 'community'"; \
		exit 2; \
	fi
	@mapfile -t run_dirs < <(find runs -mindepth 2 -maxdepth 2 -type d \
		-path "runs/$(MODEL)-*/*-v*" | sort); \
	if (( $${#run_dirs[@]} == 0 )); then \
		echo "No completed task runs found for model '$(MODEL)'"; \
		exit 1; \
	fi; \
	for run_dir in "$${run_dirs[@]}"; do \
		echo "Publishing $$run_dir as $(STATUS)"; \
		$(PYTHON) neb results publish "$$run_dir" --status "$(STATUS)" \
			--skip-existing || exit $$?; \
	done
	@$(MAKE) --no-print-directory export

publish-verified: ## Publish all runs for MODEL as maintainer-verified
	@$(MAKE) --no-print-directory publish MODEL="$(MODEL)" STATUS=verified

publish-community: ## Publish all runs for MODEL as community-unverified
	@$(MAKE) --no-print-directory publish MODEL="$(MODEL)" STATUS=community

package: ## Build Python source and wheel distributions
	uv build

site-install: ## Install locked dashboard dependencies
	npm ci --prefix $(SITE_DIR)

site-test: ## Run dashboard component and accessibility tests
	npm test --prefix $(SITE_DIR)

site-build: export ## Build the static dashboard (set BASE_PATH for Pages)
	ASTRO_TELEMETRY_DISABLED=1 BASE_PATH=$(BASE_PATH) npm run build --prefix $(SITE_DIR)

site-check: site-test site-build audit ## Test, build, and audit the dashboard

site-dev: export ## Start the Astro development server
	ASTRO_TELEMETRY_DISABLED=1 npm run dev --prefix $(SITE_DIR)

audit: ## Audit all dashboard dependencies
	npm audit --prefix $(SITE_DIR)

clean: ## Remove generated build and cache directories
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	rm -rf $(SITE_DIR)/dist $(SITE_DIR)/.astro
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
