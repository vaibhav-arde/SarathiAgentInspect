"""Framework-wide constants.

Centralizes magic strings, default paths, and well-known values
so they are never scattered across the codebase.
"""

from pathlib import Path
from typing import Final

# ── Package Metadata ──────────────────────────────────────────────
FRAMEWORK_NAME: Final[str] = "SarathiAgentInspect"
FRAMEWORK_VERSION: Final[str] = "0.1.0"

# ── Environment Names ────────────────────────────────────────────
ENV_LOCAL: Final[str] = "local"
ENV_CI: Final[str] = "ci"
ENV_PRODUCTION: Final[str] = "production"
SUPPORTED_ENVIRONMENTS: Final[frozenset[str]] = frozenset({ENV_LOCAL, ENV_CI, ENV_PRODUCTION})

# ── Environment Variable Keys ────────────────────────────────────
ENV_VAR_SARATHI_ENV: Final[str] = "SARATHI_ENV"
ENV_VAR_SARATHI_DEBUG: Final[str] = "SARATHI_DEBUG"
ENV_VAR_SARATHI_LOG_LEVEL: Final[str] = "SARATHI_LOG_LEVEL"

# ── Default Paths ────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_FILE: Final[str] = "default.yaml"
DATASETS_DIR: Final[Path] = PROJECT_ROOT / "datasets"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
CACHE_DIR: Final[Path] = PROJECT_ROOT / ".cache"

# ── Default Timeouts (seconds) ───────────────────────────────────
DEFAULT_LLM_TIMEOUT: Final[int] = 120
DEFAULT_API_TIMEOUT: Final[int] = 30
DEFAULT_HEALTH_CHECK_TIMEOUT: Final[int] = 10

# ── Retry Defaults ───────────────────────────────────────────────
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
DEFAULT_MAX_DELAY: Final[int] = 60
DEFAULT_JITTER: Final[bool] = True

# ── Evaluation Defaults ──────────────────────────────────────────
DEFAULT_THRESHOLD: Final[float] = 0.7
DEFAULT_MAX_WORKERS: Final[int] = 4
DEFAULT_BATCH_SIZE: Final[int] = 10

# ── Supported Providers ──────────────────────────────────────────
PROVIDER_OLLAMA: Final[str] = "ollama"
PROVIDER_OPENAI: Final[str] = "openai"
PROVIDER_ANTHROPIC: Final[str] = "anthropic"
PROVIDER_GEMINI: Final[str] = "gemini"
PROVIDER_AZURE_OPENAI: Final[str] = "azure_openai"
PROVIDER_BEDROCK: Final[str] = "bedrock"
