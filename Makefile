.PHONY: install test lint run docker-up docker-down k8s-apply clean help

install:      ## Create .venv and install dev dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

test:         ## Unit + end-to-end tests on synthetic data (SQLite, no Docker)
	.venv/bin/python -m pytest -q

lint:         ## Static checks
	.venv/bin/ruff check src tests run_pipeline.py

run:          ## Run the pipeline against the database in .env
	.venv/bin/python run_pipeline.py

docker-up:    ## Build and run PostgreSQL + pipeline once
	docker compose up --build

docker-down:  ## Stop containers (keeps the database volume)
	docker compose down

k8s-apply:    ## Apply namespace, postgres and the pipeline Job (see k8s/README.md)
	kubectl apply -f k8s/namespace.yaml && kubectl apply -f k8s/postgres.yaml && kubectl apply -f k8s/job.yaml

clean:        ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*##' Makefile | awk 'BEGIN{FS=":.*## "}{printf "%-12s %s\n",$$1,$$2}'
