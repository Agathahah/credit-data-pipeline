# ============================================================
# Makefile — shortcut untuk perintah Docker yang sering dipakai
# Cara pakai: ketik 'make <target>' di terminal
# Contoh: make build, make run, make logs
# ============================================================

.PHONY: build run stop logs clean db-only pipeline-only ps help

# Build Docker image dari Dockerfile
build:
	docker-compose build

# Jalankan full stack (postgres + pipeline)
run:
	docker-compose up

# Jalankan di background (detached mode)
run-detached:
	docker-compose up -d

# Hanya jalankan PostgreSQL (untuk development)
db-only:
	docker-compose up postgres

# Hanya jalankan pipeline (PostgreSQL harus sudah jalan)
pipeline-only:
	docker-compose up pipeline

# Lihat logs semua service
logs:
	docker-compose logs -f

# Lihat logs hanya pipeline
logs-pipeline:
	docker-compose logs -f pipeline

# Stop semua container
stop:
	docker-compose down

# Stop dan hapus semua data (HATI-HATI: data PostgreSQL ikut terhapus!)
clean:
	docker-compose down -v
	docker rmi credit-data-pipeline-docker_pipeline 2>/dev/null || true

# Lihat status container yang sedang berjalan
ps:
	docker-compose ps

# Masuk ke dalam container pipeline (untuk debugging)
shell:
	docker-compose exec pipeline /bin/bash

# Masuk ke PostgreSQL via psql (untuk debugging database)
psql:
	docker-compose exec postgres psql -U dataengineer -d credit_risk_db

# Test koneksi database dari dalam pipeline container
test-db:
	docker-compose exec pipeline python -c "\
		from src.utils.db import get_engine; \
		engine = get_engine(); \
		print('✅ Database connection OK:', engine.url)"

# Help
help:
	@echo "Perintah yang tersedia:"
	@echo "  make build          - Build Docker image"
	@echo "  make run            - Jalankan pipeline lengkap"
	@echo "  make run-detached   - Jalankan di background"
	@echo "  make db-only        - Hanya jalankan PostgreSQL"
	@echo "  make logs           - Lihat semua logs"
	@echo "  make logs-pipeline  - Lihat logs pipeline saja"
	@echo "  make stop           - Stop semua container"
	@echo "  make clean          - Stop dan hapus semua data"
	@echo "  make ps             - Status container"
	@echo "  make shell          - Masuk ke container pipeline"
	@echo "  make psql           - Masuk ke PostgreSQL"
	@echo "  make test-db        - Test koneksi database"
