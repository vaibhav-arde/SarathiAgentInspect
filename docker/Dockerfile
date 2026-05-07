# ──────────────────────────────────────────────
# SarathiAgentInspect — Multi-stage Dockerfile
# Stage 1: Builder (UV + dependency installation)
# Stage 2: Runtime (slim image + application)
# ──────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────
FROM python:3.13-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set workdir
WORKDIR /app

# Copy project files needed for dependency resolution
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install dependencies (production only)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# ── Stage 2: Runtime ─────────────────────────
FROM python:3.13-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 sarathi && \
    useradd --uid 1000 --gid sarathi --shell /bin/bash --create-home sarathi

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Set PATH to use virtualenv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV SARATHI_ENV="production"

# Create directories for datasets and reports
RUN mkdir -p /app/datasets /app/reports /app/.cache && \
    chown -R sarathi:sarathi /app

# Switch to non-root user
USER sarathi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sarathi_agent_inspect import __version__; print(__version__)" || exit 1

# Default entrypoint — run evaluations via pytest
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["--help"]
