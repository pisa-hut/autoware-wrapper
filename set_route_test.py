#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import autoware_adapi_v1_msgs.srv as autoware_adapi_v1_msgs_srv
import geometry_msgs.msg as geometry_msgs
import rclpy
from rclpy.node import Node

# ------------------------------------------------------------------
# Quick edit area: change these values in code (no CLI needed)
# ------------------------------------------------------------------
FRAME_ID = "map"
TIMEOUT_SEC = 30.0
ALLOW_GOAL_MODIFICATION = False


@dataclass(frozen=True)
class PoseXYZH:
    x: float
    y: float
    z: float
    h: float  # yaw in radians


START_POSE = PoseXYZH(
    x=160.2,
    y=-63.7,
    z=3,
    h=3.07,
)

GOAL_POSE = PoseXYZH(
    x=126.694,
    y=-92.9380,
    z=0.0,
    h=5.39947,
)


def yaw_to_quat(yaw_rad: float) -> tuple[float, float]:
    return math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)


def wait_for_service(node: Node, client, name: str, timeout_sec: float) -> None:
    start = time.time()
    while not client.wait_for_service(timeout_sec=1.0):
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Service {name} not available after {timeout_sec}s")
        node.get_logger().info(f"Waiting for service {name}...")
    node.get_logger().info(f"Service {name} is available.")


def wait_future(node: Node, fut, timeout_sec: float, op_name: str):
    start = time.time()
    while rclpy.ok() and not fut.done() and time.time() - start < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not fut.done():
        raise TimeoutError(f"{op_name} response timeout after {timeout_sec}s")
    return fut.result()


def call_initialize_localization(
    node: Node,
    client,
    start_pose: PoseXYZH,
    frame_id: str,
    timeout_sec: float,
) -> None:
    req = autoware_adapi_v1_msgs_srv.InitializeLocalization.Request()
    pose_msg = geometry_msgs.PoseWithCovarianceStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = frame_id

    pose_msg.pose.pose.position.x = float(start_pose.x)
    pose_msg.pose.pose.position.y = float(start_pose.y)
    pose_msg.pose.pose.position.z = float(start_pose.z)
    qz, qw = yaw_to_quat(start_pose.h)
    pose_msg.pose.pose.orientation.z = qz
    pose_msg.pose.pose.orientation.w = qw

    pose_msg.pose.covariance = [0.0] * 36
    sigma_pos = 1e-3
    sigma_ang = 1e-4
    pose_msg.pose.covariance[0] = sigma_pos**2
    pose_msg.pose.covariance[7] = sigma_pos**2
    pose_msg.pose.covariance[14] = sigma_pos**2
    pose_msg.pose.covariance[21] = sigma_ang**2
    pose_msg.pose.covariance[28] = sigma_ang**2
    pose_msg.pose.covariance[35] = sigma_ang**2

    req.pose = [pose_msg]

    node.get_logger().info(
        f"Calling InitializeLocalization with start x={start_pose.x}, y={start_pose.y}, z={start_pose.z}, h={start_pose.h}"
    )
    fut = client.call_async(req)
    res = wait_future(node, fut, timeout_sec, "InitializeLocalization")
    if res is None or not res.status.success:
        status_msg = getattr(res.status, "message", None) if res else "no response"
        code = getattr(res.status, "code", "unknown") if res else "no response"
        succ = getattr(res.status, "success", "unknown") if res else "no response"
        raise RuntimeError(
            f"InitializeLocalization failed: code={code}, success={succ}, message={status_msg}"
        )
    node.get_logger().info(
        f"InitializeLocalization success: code={res.status.code}, message={res.status.message}"
    )


def call_clear_route(node: Node, client, timeout_sec: float) -> None:
    req = autoware_adapi_v1_msgs_srv.ClearRoute.Request()
    node.get_logger().info("Calling ClearRoute")
    fut = client.call_async(req)
    res = wait_future(node, fut, timeout_sec, "ClearRoute")
    if res is None or not res.status.success:
        status_msg = getattr(res.status, "message", None) if res else "no response"
        code = getattr(res.status, "code", "unknown") if res else "no response"
        succ = getattr(res.status, "success", "unknown") if res else "no response"
        raise RuntimeError(f"ClearRoute failed: code={code}, success={succ}, message={status_msg}")
    node.get_logger().info(
        f"ClearRoute success: code={res.status.code}, message={res.status.message}"
    )


def call_set_route_points(
    node: Node,
    client,
    goal_pose: PoseXYZH,
    frame_id: str,
    allow_goal_modification: bool,
    timeout_sec: float,
) -> None:
    req = autoware_adapi_v1_msgs_srv.SetRoutePoints.Request()
    req.header.frame_id = frame_id
    req.header.stamp = node.get_clock().now().to_msg()

    goal = geometry_msgs.Pose()
    goal.position.x = float(goal_pose.x)
    goal.position.y = float(goal_pose.y)
    goal.position.z = float(goal_pose.z)
    qz, qw = yaw_to_quat(goal_pose.h)
    goal.orientation.z = qz
    goal.orientation.w = qw

    req.goal = goal
    req.waypoints = []
    req.option.allow_goal_modification = bool(allow_goal_modification)

    node.get_logger().info(
        f"Calling SetRoutePoints with goal x={goal_pose.x}, y={goal_pose.y}, z={goal_pose.z}, h={goal_pose.h}"
    )
    fut = client.call_async(req)
    res = wait_future(node, fut, timeout_sec, "SetRoutePoints")
    if res is None or not res.status.success:
        status_msg = getattr(res.status, "message", None) if res else "no response"
        code = getattr(res.status, "code", "unknown") if res else "no response"
        succ = getattr(res.status, "success", "unknown") if res else "no response"
        raise RuntimeError(
            f"SetRoutePoints failed: code={code}, success={succ}, message={status_msg}"
        )

    node.get_logger().info(
        f"SetRoutePoints success: code={res.status.code}, message={res.status.message}"
    )


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("set_route_points_test")
    try:
        client_init = node.create_client(
            autoware_adapi_v1_msgs_srv.InitializeLocalization,
            "/api/localization/initialize",
        )
        client_clear = node.create_client(
            autoware_adapi_v1_msgs_srv.ClearRoute,
            "/api/routing/clear_route",
        )
        client_set_route = node.create_client(
            autoware_adapi_v1_msgs_srv.SetRoutePoints,
            "/api/routing/set_route_points",
        )

        wait_for_service(node, client_init, "InitializeLocalization", TIMEOUT_SEC)
        wait_for_service(node, client_clear, "ClearRoute", TIMEOUT_SEC)
        wait_for_service(node, client_set_route, "SetRoutePoints", TIMEOUT_SEC)

        call_initialize_localization(
            node=node,
            client=client_init,
            start_pose=START_POSE,
            frame_id=FRAME_ID,
            timeout_sec=TIMEOUT_SEC,
        )
        input(
            "Initialization done. Please start the route planner in Autoware and press Enter to continue..."
        )
        call_clear_route(
            node=node,
            client=client_clear,
            timeout_sec=TIMEOUT_SEC,
        )

        call_set_route_points(
            node=node,
            client=client_set_route,
            goal_pose=GOAL_POSE,
            frame_id=FRAME_ID,
            allow_goal_modification=ALLOW_GOAL_MODIFICATION,
            timeout_sec=TIMEOUT_SEC,
        )

        node.get_logger().info("All done: init pose + set route succeeded.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
