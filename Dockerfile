# ── Build Stage ────────────────────────────────
FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install UV
COPY --from=astral-sh/setup-uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ── Runtime Stage ──────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy source code
COPY . .

# Set environment variables
ENV SARATHI_ENV=prod

# Default command
ENTRYPOINT ["pytest"]
CMD ["tests/", "-m", "smoke"]
