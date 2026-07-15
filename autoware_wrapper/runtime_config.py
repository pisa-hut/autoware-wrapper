from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeTimeouts:
    service_discovery_timeout_sec: float
    service_response_timeout_sec: float
    localization_timeout_sec: float
    planning_timeout_sec: float
    engage_timeout_sec: float
    stop_timeout_sec: float
    route_clear_timeout_sec: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


_TIMEOUT_DEFAULTS = RuntimeTimeouts(
    service_discovery_timeout_sec=30.0,
    service_response_timeout_sec=10.0,
    localization_timeout_sec=30.0,
    planning_timeout_sec=60.0,
    engage_timeout_sec=30.0,
    stop_timeout_sec=10.0,
    route_clear_timeout_sec=10.0,
)


def resolve_runtime_timeouts(runtime_cfg: dict[str, Any]) -> RuntimeTimeouts:
    """Resolve semantic timeouts, using legacy timeout_sec as a fallback when present."""
    legacy_timeout = float(runtime_cfg["timeout_sec"]) if "timeout_sec" in runtime_cfg else None

    def resolve(name: str) -> float:
        default = getattr(_TIMEOUT_DEFAULTS, name)
        fallback = legacy_timeout if legacy_timeout is not None else default
        return float(runtime_cfg.get(name, fallback))

    return RuntimeTimeouts(
        service_discovery_timeout_sec=resolve("service_discovery_timeout_sec"),
        service_response_timeout_sec=resolve("service_response_timeout_sec"),
        localization_timeout_sec=resolve("localization_timeout_sec"),
        planning_timeout_sec=resolve("planning_timeout_sec"),
        engage_timeout_sec=resolve("engage_timeout_sec"),
        stop_timeout_sec=resolve("stop_timeout_sec"),
        route_clear_timeout_sec=resolve("route_clear_timeout_sec"),
    )
