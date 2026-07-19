# ============================================================
# Makefile — Developer convenience commands
# All commands run inside Docker; no local Python install needed.
# Usage: make <target>
# ============================================================

.PHONY: dev test ingest logs shell stop clean rebuild

# Start all services (app + chroma) with live reload
dev:
	@echo "Starting ai_gen_platform..."
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up --build

# Run all tests inside the app container
test:
	docker compose run --rm app pytest tests/ -v --tb=short

# Run document ingestion (chunks, embeds, stores sample docs in ChromaDB)
ingest:
	docker compose run --rm app python scripts/ingest_documents.py

# Tail application logs (Ctrl+C to stop)
logs:
	docker compose logs -f app

# Open a bash shell inside the running app container
shell:
	docker compose exec app bash

# Stop all services
stop:
	docker compose down

# Stop and remove volumes (WARNING: destroys ChromaDB index!)
clean:
	docker compose down -v

# Force rebuild of all images (useful after requirements.txt changes)
rebuild:
	docker compose build --no-cache
	docker compose up

# Rebuild the vector index from scratch
rebuild-index:
	docker compose run --rm app python scripts/rebuild_index.py

# Seed sample users/tokens for local dev testing
seed:
	docker compose run --rm app python scripts/seed_sample_data.py

# Show running containers and their health status
status:
	docker compose ps
