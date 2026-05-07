# SarathiAgentInspect

**Enterprise-grade AI Evaluation Framework** for LLM, Chatbot, AI Agent, and RAG testing.

Built with Python 3.13, DeepEval, Pydantic, and production-grade tooling.

[![CI](https://github.com/vaibhav-arde/SarathiAgentInspect/actions/workflows/ci.yml/badge.svg)](https://github.com/vaibhav-arde/SarathiAgentInspect/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- 🔌 **Multi-Provider LLM Support** — Ollama, OpenAI, Anthropic, Gemini, Azure OpenAI
- 📊 **50+ Evaluation Metrics** — Via DeepEval (Faithfulness, Hallucination, GEval, etc.)
- 🤖 **Agent Evaluation** — Tool calling, planning, memory, multi-step reasoning
- 📚 **RAG Evaluation** — Retriever, generator, context quality, citation validation
- 🛡️ **Safety Testing** — Toxicity, bias, prompt injection detection
- 📈 **Observability** — Langfuse integration for tracing and monitoring
- 🐳 **Docker Ready** — Multi-stage builds with Docker Compose stack
- 🔄 **CI/CD Integration** — GitHub Actions with quality gates
- ⚡ **Async Architecture** — Built for parallel execution and scalability

## Quick Start

### Prerequisites

- Python 3.13+
- [UV package manager](https://docs.astral.sh/uv/)
- Docker Desktop (optional, for Ollama/Langfuse)

### Installation

```bash
# Clone the repository
git clone https://github.com/vaibhav-arde/SarathiAgentInspect.git
cd SarathiAgentInspect

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
make install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Run Tests

```bash
# Run unit tests
make test

# Run all tests
make test-all

# Run with coverage
make test-cov
```

### Docker Stack

```bash
# Start Ollama + Langfuse stack
make docker-up

# Pull the LLM model
docker exec sarathi-ollama ollama pull gemma4:31b-cloud

# Stop the stack
make docker-down
```

## Development

### Available Make Targets

| Command | Description |
|---|---|
| `make install` | Install all dependencies |
| `make lint` | Run Ruff linter with auto-fix |
| `make format` | Format code with Black + Ruff |
| `make typecheck` | Run mypy type checking |
| `make test` | Run unit tests |
| `make test-all` | Run all tests (unit + integration) |
| `make test-cov` | Run tests with coverage report |
| `make ci` | Full CI pipeline (lint + typecheck + test) |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start Docker Compose stack |
| `make docker-down` | Stop Docker Compose stack |
| `make clean` | Clean caches and build artifacts |

### Configuration

Configuration follows a layered resolution strategy:

```
Environment Variables → .env file → {env}.yaml → default.yaml
```

Set the environment via `SARATHI_ENV`:
```bash
export SARATHI_ENV=local    # local | ci | production
```

### Project Structure

```
src/sarathi_agent_inspect/
├── core/           # Foundation (config, logging, retry, exceptions)
├── providers/      # LLM provider implementations
├── evaluators/     # Evaluation pipelines
├── metrics/        # Metric computations
└── datasets/       # Dataset management
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

**VaibhaV Arde** — [Sarathi AI Labs](https://github.com/vaibhav-arde)