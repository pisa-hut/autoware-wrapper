from __future__ import annotations

from typing import Any

from pisa_api.av import InitResponse

AUTOWARE_COMPONENT_NAME = "autoware"


def init_response(effective_config: dict[str, Any]) -> InitResponse:
    return InitResponse(
        name=AUTOWARE_COMPONENT_NAME,
        metadata={"effective_config": effective_config},
    )
