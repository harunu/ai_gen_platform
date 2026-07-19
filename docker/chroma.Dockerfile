# ============================================================
# chroma.Dockerfile — ChromaDB vector database image
# ChromaDB publishes an official image; we use it directly.
# This Dockerfile exists for documentation and future customization.
# ============================================================

FROM chromadb/chroma:0.5.0

# ChromaDB stores its data in /chroma/chroma by default.
# We expose this as a volume in docker-compose.yml for persistence.
# Without a named volume, all indexed documents are lost on container restart.

# Expose ChromaDB HTTP port
EXPOSE 8000

# The official image already sets the CMD; no override needed.
# ChromaDB reads IS_PERSISTENT and PERSIST_DIRECTORY from environment.
