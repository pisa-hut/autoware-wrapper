from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("autoware_adapi_v1_msgs")
sys.path.insert(0, str(Path(__file__).parents[1] / "autoware_wrapper"))
autoware = importlib.import_module("autoware")


def _wrapper(runtime: dict[str, object] | None = None):
    return autoware.AutowarePureAV(cfg={"autoware": {"runtime": runtime or {}}})


def test_legacy_timeout_warns_once_and_effective_config_is_resolved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    wrapper = _wrapper({"timeout_sec": 12.5, "planning_timeout_sec": 45.0})

    runtime = wrapper._effective_config()["runtime"]
    assert "timeout_sec" not in runtime
    assert runtime["service_discovery_timeout_sec"] == 12.5
    assert runtime["planning_timeout_sec"] == 45.0
    assert "autoware.runtime.timeout_sec is deprecated" in caplog.text

    wrapper._configure(".", {"autoware": {"runtime": {"timeout_sec": 8.0}}})
    assert caplog.text.count("autoware.runtime.timeout_sec is deprecated") == 1


def test_reset_state_stages_use_their_semantic_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(
        {
            "localization_timeout_sec": 11.0,
            "planning_timeout_sec": 22.0,
            "engage_timeout_sec": 33.0,
        }
    )
    state_waits: list[tuple[float, str]] = []
    state_change_waits: list[tuple[float, str]] = []
    engage_messages: list[str] = []

    monkeypatch.setattr(wrapper, "_call_initialize_localization", lambda: None)
    monkeypatch.setattr(wrapper, "_call_set_route_points", lambda _sps: None)
    monkeypatch.setattr(autoware.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        wrapper,
        "_wait_for_autoware_state",
        lambda _state, _message, timeout, key: state_waits.append((timeout, key)),
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_for_autoware_state_to_change",
        lambda _state, _message, timeout, key: state_change_waits.append((timeout, key)),
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_for_engage_ready_stable",
        lambda message: engage_messages.append(message),
    )

    wrapper._initialize_localization_for_reset()
    wrapper._set_route_for_reset(SimpleNamespace())
    wrapper._wait_for_planning_ready()

    assert state_waits == [
        (11.0, "localization_timeout_sec"),
        (22.0, "planning_timeout_sec"),
    ]
    assert state_change_waits == [(22.0, "planning_timeout_sec")]
    assert engage_messages == ["Autoware ready to engage timed out."]
    assert wrapper._timeouts.engage_timeout_sec == 33.0


def test_service_wait_errors_identify_their_timeout_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(
        {
            "service_discovery_timeout_sec": 0.0,
            "service_response_timeout_sec": 0.0,
        }
    )
    unavailable_client = SimpleNamespace(wait_for_service=lambda timeout_sec: False)
    unfinished_future = SimpleNamespace(done=lambda: False)
    monkeypatch.setattr(autoware.rclpy, "ok", lambda: True)

    with pytest.raises(autoware.AvTimeout, match="service_discovery_timeout_sec"):
        wrapper._wait_for_service(unavailable_client, "ExampleService")

    with pytest.raises(autoware.AvTimeout, match="service_response_timeout_sec"):
        wrapper._wait_for_service_response(unfinished_future, "ExampleService")


def test_stop_and_route_clear_timeouts_remain_best_effort(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper({"stop_timeout_sec": 0.0, "route_clear_timeout_sec": 0.0})
    monkeypatch.setattr(wrapper, "_call_change_to_stop", lambda: None)
    monkeypatch.setattr(wrapper, "_call_clear_route", lambda: None)
    monkeypatch.setattr(autoware.rclpy, "ok", lambda: True)
    monkeypatch.setattr(autoware.time, "sleep", lambda _seconds: None)
    caplog.set_level(logging.WARNING)

    wrapper._stop_autoware_vehicle()

    assert "autoware.runtime.stop_timeout_sec" in caplog.text
    assert "autoware.runtime.route_clear_timeout_sec" in caplog.text
