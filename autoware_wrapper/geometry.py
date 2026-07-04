from __future__ import annotations

import math
from dataclasses import replace

from pisa_api.av import (
    ObjectKinematicData,
    ObjectStateData,
    ObservationData,
    ShapeCenterPoseData,
    ShapeData,
    ShapeType,
)


class ObservationContractError(ValueError):
    """An observation violates the normative sim-core data contract."""


class ObservationNormalizer:
    """Validate observations and retain episode-local static shape metadata."""

    def __init__(self) -> None:
        self._shapes_by_tracking_id: dict[int, ShapeData] = {}

    def reset(self) -> None:
        self._shapes_by_tracking_id.clear()

    def normalize(
        self,
        observation: ObservationData,
        timestamp_ns: int,
    ) -> tuple[ObjectStateData, list[ObjectStateData]]:
        _validate_state(observation.ego, timestamp_ns, require_shape=False, label="ego")
        states: list[ObjectStateData] = []
        for index, agent in enumerate(observation.agents):
            label = f"agent[{index}]"
            state = agent.state
            shape = state.shape
            if shape is None:
                if agent.tracking_id is None:
                    raise ObservationContractError(
                        f"{label} has no shape and no tracking_id for geometry lookup"
                    )
                shape = self._shapes_by_tracking_id.get(agent.tracking_id)
                if shape is None:
                    raise ObservationContractError(
                        f"{label} first observation is missing shape metadata"
                    )
                state = replace(state, shape=shape)
            elif agent.tracking_id is not None:
                cached = self._shapes_by_tracking_id.get(agent.tracking_id)
                if cached is not None and not _shapes_equivalent(cached, shape):
                    raise ObservationContractError(
                        f"{label} changed static shape for tracking_id={agent.tracking_id}"
                    )
                self._shapes_by_tracking_id[agent.tracking_id] = shape

            _validate_state(state, timestamp_ns, require_shape=True, label=label)
            states.append(state)
        return observation.ego, states


def compose_shape_center_pose(
    kinematic: ObjectKinematicData,
    center: ShapeCenterPoseData,
) -> tuple[float, float, float, float, float, float, float]:
    """Compose the canonical actor pose with its actor-local shape-center pose."""
    cos_yaw = math.cos(kinematic.yaw)
    sin_yaw = math.sin(kinematic.yaw)
    x = kinematic.x + cos_yaw * center.x - sin_yaw * center.y
    y = kinematic.y + sin_yaw * center.x + cos_yaw * center.y
    z = kinematic.z + center.z

    local = _rpy_to_quaternion(center.roll, center.pitch, center.yaw)
    parent = (0.0, 0.0, math.sin(kinematic.yaw / 2.0), math.cos(kinematic.yaw / 2.0))
    qx, qy, qz, qw = _multiply_quaternions(parent, local)
    return x, y, z, qx, qy, qz, qw


def validate_ackermann_payload(payload: dict[str, float]) -> None:
    required = {"steer", "speed"}
    allowed = required | {"steer_speed", "acceleration", "jerk"}
    if not required <= payload.keys() or not payload.keys() <= allowed:
        raise ValueError("ACKERMANN payload contains missing or unknown fields")
    for name, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"ACKERMANN {name} must be a finite number")
    if payload["speed"] < 0:
        raise ValueError("ACKERMANN speed must be non-negative")
    if payload.get("steer_speed", 0.0) < 0:
        raise ValueError("ACKERMANN steer_speed must be non-negative")


def _validate_state(
    state: ObjectStateData,
    timestamp_ns: int,
    *,
    require_shape: bool,
    label: str,
) -> None:
    if state.kinematic.time_ns != timestamp_ns:
        raise ObservationContractError(
            f"{label} kinematic time_ns={state.kinematic.time_ns} does not match {timestamp_ns}"
        )
    _require_finite(
        label,
        state.kinematic.x,
        state.kinematic.y,
        state.kinematic.z,
        state.kinematic.yaw,
        state.kinematic.speed,
        state.kinematic.acceleration,
        state.kinematic.yaw_rate,
        state.kinematic.yaw_acceleration,
    )
    if state.shape is None:
        if require_shape:
            raise ObservationContractError(f"{label} is missing required shape metadata")
        return

    shape = state.shape
    _require_finite(
        f"{label}.shape",
        shape.dimensions.x,
        shape.dimensions.y,
        shape.dimensions.z,
        shape.center.x,
        shape.center.y,
        shape.center.z,
        shape.center.roll,
        shape.center.pitch,
        shape.center.yaw,
    )
    if shape.type == ShapeType.BOUNDING_BOX and (
        shape.dimensions.x <= 0 or shape.dimensions.y <= 0 or shape.dimensions.z <= 0
    ):
        raise ObservationContractError(f"{label} shape dimensions must be positive")
    for vertex_index, vertex in enumerate(shape.vertices):
        _require_finite(f"{label}.shape.vertices[{vertex_index}]", vertex.x, vertex.y, vertex.z)


def _require_finite(label: str, *values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ObservationContractError(f"{label} contains a non-finite numeric value")


def _shapes_equivalent(left: ShapeData, right: ShapeData) -> bool:
    """Compare repeated static geometry with a documented 1e-9 tolerance."""
    if (
        left.type != right.type
        or left.reference_point != right.reference_point
        or len(left.vertices) != len(right.vertices)
    ):
        return False
    left_values = (
        left.dimensions.x,
        left.dimensions.y,
        left.dimensions.z,
        left.center.x,
        left.center.y,
        left.center.z,
        left.center.roll,
        left.center.pitch,
        left.center.yaw,
    )
    right_values = (
        right.dimensions.x,
        right.dimensions.y,
        right.dimensions.z,
        right.center.x,
        right.center.y,
        right.center.z,
        right.center.roll,
        right.center.pitch,
        right.center.yaw,
    )
    if not all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(left_values, right_values, strict=True)
    ):
        return False
    return all(
        math.isclose(a.x, b.x, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(a.y, b.y, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(a.z, b.z, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(left.vertices, right.vertices, strict=True)
    )


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _multiply_quaternions(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
