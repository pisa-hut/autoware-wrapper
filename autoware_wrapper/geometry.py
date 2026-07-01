from __future__ import annotations

import math

from pisa_api.av import ObjectKinematicData, ObjectStateData, ObservationData, ShapeCenterPoseData


def states_from_observation(
    observation: ObservationData,
) -> tuple[ObjectStateData, list[ObjectStateData]]:
    """Strip optional agent identity metadata at the Autoware boundary."""
    return observation.ego, [agent.state for agent in observation.agents]


def compose_shape_center_pose(
    kinematic: ObjectKinematicData,
    center: ShapeCenterPoseData,
    *,
    yaw_sign: float = 1.0,
    yaw_offset_rad: float = 0.0,
) -> tuple[float, float, float, float, float, float, float]:
    """Return the shape-center world position and ROS quaternion.

    ``Shape.center`` is expressed in the local frame rooted at the kinematic
    pose. The kinematic contract currently carries yaw only, while the center
    metadata may carry a complete local orientation.
    """
    parent_yaw = math.atan2(
        math.sin(yaw_sign * kinematic.yaw + yaw_offset_rad),
        math.cos(yaw_sign * kinematic.yaw + yaw_offset_rad),
    )
    cos_yaw = math.cos(parent_yaw)
    sin_yaw = math.sin(parent_yaw)
    x = kinematic.x + cos_yaw * center.x - sin_yaw * center.y
    y = kinematic.y + sin_yaw * center.x + cos_yaw * center.y
    z = kinematic.z + center.z

    local = _rpy_to_quaternion(center.roll, center.pitch, yaw_sign * center.yaw)
    parent = (0.0, 0.0, math.sin(parent_yaw / 2.0), math.cos(parent_yaw / 2.0))
    qx, qy, qz, qw = _multiply_quaternions(parent, local)
    return x, y, z, qx, qy, qz, qw


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
