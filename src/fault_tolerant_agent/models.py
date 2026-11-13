"""Chat model construction for Azure AI Foundry deployments (Entra ID auth, no API keys)."""

from __future__ import annotations

from langchain_core.rate_limiters import InMemoryRateLimiter

from .config import Settings


def make_rate_limiter(s: Settings) -> InMemoryRateLimiter:
    """Proactive throttling. Limits *requests* only (not tokens) and lives in this process only;
    for a fleet of workers use a shared gateway or queue instead."""
    return InMemoryRateLimiter(
        requests_per_second=s.requests_per_second,
        check_every_n_seconds=s.rate_check_every_n_seconds,
        max_bucket_size=s.rate_max_bucket_size,
    )


def make_azure_model(deployment: str, s: Settings):
    """One Foundry deployment, with its own rate limiter and a request timeout.

    ``max_retries`` defaults to the SDK's own retry loop (6!). It is set to ``s.sdk_max_retries``
    (0) so that ModelRetryMiddleware is the single, visible retry layer.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI

    if not s.endpoint:
        raise RuntimeError("AZURE_AI_ENDPOINT is not set (run infra/provision.ps1 or create .env)")
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureChatOpenAI(
        azure_endpoint=s.endpoint,
        azure_deployment=deployment,
        api_version=s.api_version,
        azure_ad_token_provider=token_provider,
        rate_limiter=make_rate_limiter(s),
        timeout=s.request_timeout,
        max_retries=s.sdk_max_retries,
        max_tokens=s.max_tokens,
    )
