"""Provider abstractions, HTTP retry/backoff client, and error types."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import requests

from squeezeradar.config import ProviderConfig
from squeezeradar.data import SymbolDataset, TTLCache

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Base error for market data providers."""


class ProviderBlockedError(ProviderError):
    """Raised when an upstream API rejects access (e.g., HTTP 451 or 403 geo-block)."""


class ProviderRateLimitError(ProviderError):
    """Raised when retries on HTTP 429 are exhausted."""


class ProviderUnavailableError(ProviderError):
    """Raised when the upstream cannot be reached (timeouts / connection errors after retries)."""


@dataclass(frozen=True)
class ProviderBadge:
    """Metadata describing the active data source for UI and CLI display."""

    source: str  # "binance" | "hyperliquid" | "replay"
    mode: str  # "live" | "replay"
    label: str
    detail: str
    fallback_reason: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ProviderResult:
    """Loaded datasets alongside provenance badge metadata."""

    datasets: dict[str, SymbolDataset]
    badge: ProviderBadge
    errors: dict[str, str] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Interface implemented by Binance, Hyperliquid, and Replay providers."""

    name: str

    def fetch_symbol(
        self,
        symbol: str,
        limit: int = 720,
        as_of: str | None = None,
    ) -> SymbolDataset:
        """Fetch raw and aligned data for a single perpetual symbol."""
        ...


class HttpJsonClient:
    """Requests wrapper with timeouts, 429 exponential backoff, and 451 detection."""

    def __init__(
        self,
        session: requests.Session | None = None,
        config: ProviderConfig | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.config = config or ProviderConfig()
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Execute an HTTP request and return parsed JSON with retries on 429/5xx."""
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.config.timeout_sec,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    delay = self.config.backoff_base_sec * (2**attempt)
                    logger.warning("HTTP request error (%s), retrying in %.2fs", exc, delay)
                    self.sleep_fn(delay)
                    continue
                raise ProviderUnavailableError(f"Request failed for {url}: {exc}") from exc

            if resp.status_code in (451, 403):
                raise ProviderBlockedError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
                )

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self.config.backoff_base_sec * (2**attempt)
                else:
                    delay = self.config.backoff_base_sec * (2**attempt)
                if attempt < self.config.max_retries:
                    logger.warning(
                        "HTTP %d from %s, backing off %.2fs (attempt %d/%d)",
                        resp.status_code,
                        url,
                        delay,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    self.sleep_fn(delay)
                    continue
                if resp.status_code == 429:
                    raise ProviderRateLimitError(f"HTTP 429 rate limited at {url}")
                raise ProviderError(f"HTTP {resp.status_code} server error at {url}")

            if resp.status_code >= 400:
                raise ProviderError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(f"Invalid JSON response from {url}") from exc

        raise ProviderError(f"Request failed for {url}: {last_exc}")


__all__ = [
    "ProviderError",
    "ProviderBlockedError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "ProviderBadge",
    "ProviderResult",
    "MarketDataProvider",
    "HttpJsonClient",
    "TTLCache",
]
