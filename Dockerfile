# Use the official OpenEnv base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install git for dependency management
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy your project files to the root /app
COPY . /app/

# Ensure uv is installed
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Sync dependencies (this creates the .venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy all project files (including server/ and models.py)
COPY --from=builder /app /app

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# CRITICAL: Set PYTHONPATH to /app so 'import models' works everywhere
ENV PYTHONPATH="/app"

# Expose the port for Hugging Face
EXPOSE 8000

# Health check to ensure the container is alive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server from /app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]