from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from pisa_api import av_server_pb2, empty_pb2
from pisa_api.av import AvUnavailable, InitResponse
from pisa_api.av.service import GenericAvService

from autoware_wrapper import version as wrapper_version_module
from autoware_wrapper.contract import AUTOWARE_COMPONENT_NAME, init_response


def test_wrapper_version_uses_distribution_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrapper_version_module.metadata, "version", lambda name: "9.8.7")

    assert wrapper_version_module.wrapper_version() == "9.8.7"


def test_wrapper_version_falls_back_to_checkout_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "autoware-wrapper"\nversion = "1.2.3"\n')

    def missing_distribution(_name: str) -> str:
        raise wrapper_version_module.metadata.PackageNotFoundError

    monkeypatch.setattr(wrapper_version_module.metadata, "version", missing_distribution)
    monkeypatch.setattr(wrapper_version_module, "_PYPROJECT_PATH", pyproject)

    assert wrapper_version_module.wrapper_version() == "1.2.3"


def test_init_response_identifies_autoware_and_metadata_round_trips() -> None:
    effective_config = {
        "launch": {"headless": True, "extra_args": ["rviz:=false"]},
        "runtime": {"planning_timeout_sec": 12.5, "publish_agent_objects": False},
        "vehicle": {"model": "sample_vehicle", "sensor_model": "sample_sensor_kit"},
    }

    response = init_response(effective_config)

    assert isinstance(response, InitResponse)
    assert response.name == AUTOWARE_COMPONENT_NAME == "autoware"
    assert response.metadata == {"effective_config": effective_config}
    assert "dt" not in response.metadata
    assert "map" not in response.metadata
    proto_metadata = Struct()
    json_format.ParseDict(response.metadata, proto_metadata)
    assert json_format.MessageToDict(proto_metadata) == response.metadata


def test_server_entry_point_passes_wrapper_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    av_system = object()
    fake_autoware = ModuleType("autoware")
    fake_autoware.AutowarePureAV = lambda: av_system
    fake_av = ModuleType("pisa_api.av")
    fake_av.serve_av_system = lambda system, **kwargs: calls.append((system, kwargs))
    fake_wrapper = ModuleType("pisa_api.wrapper")
    fake_wrapper.setup_logging = lambda: None
    fake_version = ModuleType("version")
    fake_version.wrapper_version = lambda: "0.3.1"
    monkeypatch.setitem(sys.modules, "autoware", fake_autoware)
    monkeypatch.setitem(sys.modules, "pisa_api.av", fake_av)
    monkeypatch.setitem(sys.modules, "pisa_api.wrapper", fake_wrapper)
    monkeypatch.setitem(sys.modules, "version", fake_version)

    runpy.run_path(
        str(Path(__file__).parents[1] / "autoware_wrapper" / "server.py"),
        run_name="__main__",
    )

    assert calls == [(av_system, {"name": "autoware-wrapper", "version": "0.3.1"})]


class _Context:
    def __init__(self) -> None:
        self.code = None

    def peer(self) -> str:
        return "test"

    def set_code(self, code: object) -> None:
        self.code = code

    def set_details(self, _details: str) -> None:
        pass


def test_generated_service_ping_and_init_smoke() -> None:
    expected = init_response({"vehicle": {"model": "sample_vehicle"}})
    av_system = SimpleNamespace(init=lambda _request: expected)
    service = GenericAvService(av_system, name="autoware-wrapper", version="0.3.1")
    context = _Context()

    pong = service.Ping(empty_pb2.Empty(), context)
    initialized = service.Init(av_server_pb2.AvServerMessages.InitRequest(), context)

    assert pong.msg == "autoware-wrapper alive"
    assert pong.name == "autoware-wrapper"
    assert pong.version == "0.3.1"
    assert initialized.name == "autoware"
    assert json_format.MessageToDict(initialized.metadata) == expected.metadata


def test_generated_service_does_not_report_success_when_init_fails() -> None:
    def fail(_request: object) -> InitResponse:
        raise AvUnavailable("Autoware is unavailable")

    service = GenericAvService(SimpleNamespace(init=fail), name="autoware-wrapper", version="0.3.1")
    context = _Context()

    initialized = service.Init(av_server_pb2.AvServerMessages.InitRequest(), context)

    assert initialized.name == ""
    assert not initialized.HasField("metadata")
    assert context.code is not None
