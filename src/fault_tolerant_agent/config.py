"""Runtime configuration, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # --- models (Azure AI Foundry deployments) ---
    endpoint: str = ""
    primary_model: str = "primary"
    fallback_model: str = "fallback"
    summary_model: str = "fallback"  # cheap model used for summarization
    api_version: str = "2024-10-21"
    max_tokens: int = 300

    # --- proactive throttling (requests only, in-memory, per process) ---
    requests_per_second: float = 1.0
    rate_check_every_n_seconds: float = 0.1
    rate_max_bucket_size: int = 2

    # --- reactive retries. Layers multiply: keep the SDK's own retries at 0 ---
    request_timeout: float = 30.0
    sdk_max_retries: int = 0
    model_retry_max: int = 2
    tool_retry_max: int = 2
    retry_initial_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_max_delay: float = 20.0
    retry_jitter: bool = True

    # --- budgets ---
    model_run_limit: int = 10
    model_thread_limit: int = 40
    tool_run_limit: int = 8
    tool_thread_limit: int = 30

    # --- context management ---
    summarize_trigger_tokens: int = 4000
    summarize_keep_messages: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        d = cls()
        return cls(
            endpoint=os.environ.get("AZURE_AI_ENDPOINT", d.endpoint),
            primary_model=os.environ.get("PRIMARY_MODEL", d.primary_model),
            fallback_model=os.environ.get("FALLBACK_MODEL", d.fallback_model),
            summary_model=os.environ.get("SUMMARY_MODEL", os.environ.get("FALLBACK_MODEL", d.summary_model)),
            api_version=os.environ.get("AZURE_API_VERSION", d.api_version),
            max_tokens=_int("MAX_TOKENS", d.max_tokens),
            requests_per_second=_float("REQUESTS_PER_SECOND", d.requests_per_second),
            request_timeout=_float("REQUEST_TIMEOUT", d.request_timeout),
            sdk_max_retries=_int("SDK_MAX_RETRIES", d.sdk_max_retries),
            model_retry_max=_int("MODEL_RETRY_MAX", d.model_retry_max),
            tool_retry_max=_int("TOOL_RETRY_MAX", d.tool_retry_max),
            retry_initial_delay=_float("RETRY_INITIAL_DELAY", d.retry_initial_delay),
            model_run_limit=_int("MODEL_RUN_LIMIT", d.model_run_limit),
            model_thread_limit=_int("MODEL_THREAD_LIMIT", d.model_thread_limit),
            tool_run_limit=_int("TOOL_RUN_LIMIT", d.tool_run_limit),
            tool_thread_limit=_int("TOOL_THREAD_LIMIT", d.tool_thread_limit),
            summarize_trigger_tokens=_int("SUMMARIZE_TRIGGER_TOKENS", d.summarize_trigger_tokens),
            summarize_keep_messages=_int("SUMMARIZE_KEEP_MESSAGES", d.summarize_keep_messages),
        )

    def fast(self) -> "Settings":
        """Variant with no backoff sleeps, for tests."""
        return replace(self, retry_initial_delay=0.0, retry_backoff_factor=0.0, retry_jitter=False)
