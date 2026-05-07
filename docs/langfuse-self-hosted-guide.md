# Self-Hosted Langfuse Deployment Guide

This guide covers deploying Langfuse for observability with SarathiAgentInspect.

## Prerequisites

- Docker Desktop installed and running
- At least 2GB free RAM
- Ports 3000 and 5432 available

## Quick Start (Docker Compose)

The easiest way to deploy Langfuse is via the included docker-compose stack:

```bash
# From the project root
make docker-up
```

This starts:
- **Langfuse Server** on `http://localhost:3000`
- **PostgreSQL 16** on port `5432`
- **Ollama** on port `11434`

## Manual Setup

### Step 1: Start PostgreSQL

```bash
docker run -d \
  --name langfuse-postgres \
  -e POSTGRES_DB=langfuse \
  -e POSTGRES_USER=langfuse \
  -e POSTGRES_PASSWORD=your_secure_password \
  -p 5432:5432 \
  -v langfuse-db:/var/lib/postgresql/data \
  postgres:16-alpine
```

### Step 2: Start Langfuse Server

```bash
docker run -d \
  --name langfuse-server \
  -e DATABASE_URL=postgresql://langfuse:your_secure_password@host.docker.internal:5432/langfuse \
  -e NEXTAUTH_SECRET=$(openssl rand -base64 32) \
  -e NEXTAUTH_URL=http://localhost:3000 \
  -e SALT=$(openssl rand -base64 32) \
  -e TELEMETRY_ENABLED=false \
  -p 3000:3000 \
  langfuse/langfuse:latest
```

### Step 3: Initial Setup

1. Open `http://localhost:3000` in your browser
2. Create an account (first user becomes admin)
3. Create a new project (e.g., "SarathiAgentInspect")
4. Go to **Settings → API Keys** and create a new API key pair

### Step 4: Configure SarathiAgentInspect

Add these values to your `.env` file:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_BASE_URL=http://localhost:3000
```

## Production Deployment

### Security Checklist

- [ ] Change `NEXTAUTH_SECRET` to a strong random value
- [ ] Change `SALT` to a strong random value
- [ ] Change PostgreSQL password to a secure password
- [ ] Enable HTTPS (use a reverse proxy like Nginx or Caddy)
- [ ] Set `NEXTAUTH_URL` to your production URL
- [ ] Restrict network access to Langfuse dashboard
- [ ] Set up database backups

### Using Docker Compose for Production

Create a `docker-compose.prod.yml`:

```yaml
services:
  langfuse-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_DB_PASSWORD}
    volumes:
      - langfuse-db-data:/var/lib/postgresql/data
    restart: always

  langfuse:
    image: langfuse/langfuse:latest
    environment:
      DATABASE_URL: postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      NEXTAUTH_URL: ${LANGFUSE_URL}
      SALT: ${LANGFUSE_SALT}
      TELEMETRY_ENABLED: "false"
    ports:
      - "3000:3000"
    depends_on:
      - langfuse-db
    restart: always

volumes:
  langfuse-db-data:
```

### Cloud Deployment Options

| Platform | Approach |
|---|---|
| AWS | ECS Fargate + RDS PostgreSQL |
| GCP | Cloud Run + Cloud SQL |
| Azure | Container Apps + Azure Database for PostgreSQL |
| DigitalOcean | App Platform + Managed PostgreSQL |

## Verification

Once Langfuse is running, verify the connection:

```python
from langfuse import Langfuse

client = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000",
)

# Should print without errors
print(client.auth_check())
```

## Troubleshooting

| Issue | Solution |
|---|---|
| Port 3000 in use | Change port mapping in docker-compose.yml |
| Database connection failed | Ensure PostgreSQL is healthy: `docker logs sarathi-langfuse-db` |
| Langfuse won't start | Check logs: `docker logs sarathi-langfuse` |
| Slow startup | First startup runs migrations — may take 30-60 seconds |
