from __future__ import annotations

from autoware_wrapper.runtime_config import RuntimeTimeouts, resolve_runtime_timeouts


def test_runtime_timeouts_use_semantic_defaults() -> None:
    assert resolve_runtime_timeouts({}) == RuntimeTimeouts(
        service_discovery_timeout_sec=30.0,
        service_response_timeout_sec=10.0,
        localization_timeout_sec=30.0,
        planning_timeout_sec=60.0,
        engage_timeout_sec=30.0,
        stop_timeout_sec=10.0,
        route_clear_timeout_sec=10.0,
    )


def test_legacy_timeout_is_fallback_for_every_semantic_timeout() -> None:
    timeouts = resolve_runtime_timeouts({"timeout_sec": 12.5})

    assert set(timeouts.as_dict().values()) == {12.5}


def test_semantic_timeout_overrides_legacy_fallback() -> None:
    timeouts = resolve_runtime_timeouts(
        {
            "timeout_sec": 12.5,
            "service_response_timeout_sec": 4,
            "planning_timeout_sec": "45.0",
        }
    )

    assert timeouts.service_discovery_timeout_sec == 12.5
    assert timeouts.service_response_timeout_sec == 4.0
    assert timeouts.localization_timeout_sec == 12.5
    assert timeouts.planning_timeout_sec == 45.0
    assert timeouts.engage_timeout_sec == 12.5
    assert timeouts.stop_timeout_sec == 12.5
    assert timeouts.route_clear_timeout_sec == 12.5


def test_runtime_timeouts_as_dict_uses_public_config_keys() -> None:
    assert resolve_runtime_timeouts({}).as_dict() == {
        "service_discovery_timeout_sec": 30.0,
        "service_response_timeout_sec": 10.0,
        "localization_timeout_sec": 30.0,
        "planning_timeout_sec": 60.0,
        "engage_timeout_sec": 30.0,
        "stop_timeout_sec": 10.0,
        "route_clear_timeout_sec": 10.0,
    }
