# ============================================================
# app.Dockerfile — FastAPI application image
# Multi-stage is overkill for a dev template; single stage keeps it simple.
# Python 3.11-slim avoids pulling in ~1GB of unnecessary system packages.
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install OS-level build dependencies.
# gcc is needed to compile some Python packages (e.g. cryptography, tokenizers).
# --no-install-recommends keeps the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer caching means this layer is only
# rebuilt when requirements.txt changes, not on every code change.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't store the pip download cache in the image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container doesn't fetch it at runtime.
# This bakes the model into the image (adds ~90MB) but eliminates cold-start delay.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
# This layer rebuilds on every source change — intentionally last.
COPY . .

# Expose the application port
EXPOSE 8000

# Healthcheck: Docker will mark the container unhealthy if this fails.
# --interval: check every 30s
# --timeout: fail if no response in 10s
# --start-period: give the app 40s to start before checking
# --retries: mark unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command: uvicorn with hot-reload for development.
# In production, remove --reload and set workers based on CPU count.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
