from __future__ import annotations

import logging
import math
import os
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import autoware_adapi_v1_msgs.msg as autoware_adapi_v1_msgs
import autoware_adapi_v1_msgs.srv as autoware_adapi_v1_msgs_srv
import autoware_control_msgs.msg as autoware_control_msgs
import autoware_perception_msgs.msg as autoware_perception_msgs
import autoware_system_msgs.msg as autoware_system_msgs
import autoware_vehicle_msgs.msg as autoware_vehicle_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import rclpy
import rosgraph_msgs.msg as rosgraph_msgs
import sensor_msgs.msg as sensor_msgs
import tier4_system_msgs.msg as tier4_system_msgs
from contract import init_response
from geometry import (
    ObservationContractError,
    ObservationNormalizer,
    clamp_ackermann_speed,
    compose_shape_center_pose,
    validate_ackermann_payload,
)
from pisa_api.av import (
    AvPreconditionFailed,
    AvTimeout,
    AvUnavailable,
    ControlCommand,
    ControlMode,
    InitRequest,
    InitResponse,
    InvalidAvRequest,
    ObjectKinematicData,
    ObjectStateData,
    ObservationData,
    ResetRequest,
    ResetResponse,
    RoadObjectType,
    ScenarioPackData,
    ShapeType,
    ShouldQuitResponse,
    StepRequest,
    StepResponse,
)
from publish_manager import PublishManager, PublishMode, TopicPublisher
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import TransformBroadcaster

CLOCK_PUB_HZ = 100.0  # Hz

AUTOWARE_NODE_PREFIXES = (
    "/adapi/",
    "/control/",
    "/localization/",
    "/map/",
    "/perception/",
    "/planning/",
    "/sensing/",
    "/system/",
    "/vehicle/",
)
AUTOWARE_NODE_NAMES = {
    "/component_state_monitor",
    "/dummy_diag_publisher",
    "/logging_diag_graph",
    "/pose_initializer",
    "/robot_state_publisher",
    "/rviz2",
}

# Mapping from RoadObjectType to Autoware's ObjectClassification
OBJECT_TYPE_MAP = {
    RoadObjectType.CAR: autoware_perception_msgs.ObjectClassification.CAR,
    RoadObjectType.TRUCK: autoware_perception_msgs.ObjectClassification.TRUCK,
    RoadObjectType.BUS: autoware_perception_msgs.ObjectClassification.BUS,
    RoadObjectType.TRAILER: autoware_perception_msgs.ObjectClassification.TRAILER,
    RoadObjectType.MOTORCYCLE: autoware_perception_msgs.ObjectClassification.MOTORCYCLE,
    RoadObjectType.BICYCLE: autoware_perception_msgs.ObjectClassification.BICYCLE,
    RoadObjectType.PEDESTRIAN: autoware_perception_msgs.ObjectClassification.PEDESTRIAN,
}


logger = logging.getLogger(__name__)


class AutowarePureAV:
    """
    Autoware AV adapter:
    - init(): Create ROS node, pub/sub, services
    - reset(): Relaunch Autoware, then set initial pose + route via AD API services
    - step(): Send obs (ego + agents), wait for new control, convert to Ctrl to return to simulator
    - stop(): Stop Autoware process + ROS node
    - should_quit(): Decide whether to quit based on motion state / error / process status
    """

    def __init__(self, output_base: str | Path = ".", cfg: dict | None = None):
        self._configure(output_base, cfg or {})

        # ScenarioPack
        self._sps: ScenarioPackData | None = None
        self._map_path: Path | None = None

        # —— Autoware process & ROS node status ——
        self._autoware_proc: subprocess.Popen | None = None
        self._node: Node | None = None
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None

        # pub / sub / services
        self._publish_manager = PublishManager()
        self._kinematic_state_pub = None
        self._accel_pub = None
        self._objects_pub = None
        self._dummy_pointcloud_pub = None
        self._control_mode_pub = None
        self._gear_report_pub = None
        self._steering_report_pub = None
        self._velocity_report_pub = None
        self._occupancy_grid_pub = None
        self._tf_broadcaster = None
        self._control_sub = None
        self._gear_cmd_sub = None
        self._autoware_state_sub = None
        self._diagnostics_graph_sub = None
        self._client_initial_localization = None
        self._client_set_route_points = None
        self._client_change_to_auto = None

        # Adapter internal state
        self._initialized: bool = False

        self._base_time_ns: int = 0  # time at sim_time == 0 (nanoseconds)
        self._episode_ros_offset_ns: int = 0
        self._sim_time_ns: int = 0  # time at current sim step (nanoseconds)
        self._current_ros_time_ns: int = 0  # current ROS time (nanoseconds)

        self._vehicle_state: int | None = None
        self._control_mode: int | None = autoware_vehicle_msgs.ControlModeReport.NO_COMMAND
        self._operation_mode_state = None
        self._autoware_motion_state = None
        self._route_state = None
        self._current_gear: int | None = autoware_vehicle_msgs.GearCommand.NONE
        self._latest_control: autoware_control_msgs.Control = None
        self._latest_control_stamp = 0
        self._latest_diagnostics: list[tuple[int, str, str, str]] = []
        self._kinematic: ObjectKinematicData = ObjectKinematicData()
        self._quit_flag: bool = False
        self._last_error: str | None = None
        self._agents: list[ObjectStateData] = []
        self._observation_normalizer = ObservationNormalizer()
        self._last_sim_timestamp_ns: int | None = None

    def _configure(self, output_base: str | Path, cfg: dict) -> None:
        self._output_base = Path(output_base)

        self.config = cfg
        self._autoware_cfg = cfg.get("autoware", {})

        # —— Autoware Launch Settings ——
        self._root = Path(self._autoware_cfg.get("root", "/autoware"))
        self._ros_setup_script = self._autoware_cfg.get(
            "ros_setup_script", "/opt/ros/humble/setup.bash"
        )

        launch_cfg = self._autoware_cfg.get("launch", {})
        self._launch_package = launch_cfg.get("package", "autoware_launch")
        self._launch_file = launch_cfg.get("file", "pisa.launch.xml")
        self._headless = bool(launch_cfg.get("headless", True))
        self._extra_launch_args: list[str] = list(launch_cfg.get("extra_args", []))
        self._log_enabled = bool(launch_cfg.get("log", False))
        self._autoware_log_path = self._output_base / launch_cfg.get(
            "log_path", "autoware_launch.log"
        )

        data_cfg = self._autoware_cfg.get("data", {})
        self._data_path = Path(data_cfg.get("data_path", "/autoware_data"))

        veh_cfg = self._autoware_cfg.get("vehicle", {})
        self._vehicle_model = veh_cfg.get("model", "sample_vehicle")
        self._sensor_model = veh_cfg.get("sensor_model", "sample_sensor_kit")

        self._rt_cfg = self._autoware_cfg.get("runtime", {})
        self._publish_agent_objects = bool(self._rt_cfg.get("publish_agent_objects", True))
        publish_agent_objects_env = os.getenv("PISA_PUBLISH_AGENT_OBJECTS")
        if publish_agent_objects_env is not None:
            self._publish_agent_objects = publish_agent_objects_env.lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        self._timeout_sec = float(self._rt_cfg.get("timeout_sec", 30.0))
        self._control_timeout_sec = float(self._rt_cfg.get("control_timeout_sec", 0.01))
        self._engage_ready_stable_sec = float(self._rt_cfg.get("engage_ready_stable_sec", 0.0))
        self._engage_retry_sec = float(self._rt_cfg.get("engage_retry_sec", 3.0))
        self._engage_retry_interval_sec = float(self._rt_cfg.get("engage_retry_interval_sec", 0.2))
        self._diagnostics_graph_enabled = bool(self._rt_cfg.get("diagnostics_graph_enabled", True))
        self._diagnostics_as_precondition_failure = bool(
            self._rt_cfg.get("diagnostics_as_precondition_failure", True)
        )
        self._lane_departure_boundary_check_enabled = bool(
            self._rt_cfg.get("lane_departure_boundary_check_enabled", True)
        )
        self._runtime_param_timeout_sec = float(self._rt_cfg.get("runtime_param_timeout_sec", 5.0))
        self._precondition_diagnostic_hardware_ids = set(
            self._rt_cfg.get(
                "precondition_diagnostic_hardware_ids",
                ["lane_departure_checker"],
            )
        )
        self._shutdown_interrupt_grace_sec = float(
            self._rt_cfg.get("shutdown_interrupt_grace_sec", 2.0)
        )
        self._shutdown_terminate_grace_sec = float(
            self._rt_cfg.get("shutdown_terminate_grace_sec", 1.0)
        )
        self._shutdown_kill_grace_sec = float(self._rt_cfg.get("shutdown_kill_grace_sec", 2.0))
        self._process_group_cleanup_timeout_sec = float(
            self._rt_cfg.get("process_group_cleanup_timeout_sec", 3.0)
        )
        self._ros_graph_cleanup_timeout_sec = float(
            self._rt_cfg.get("ros_graph_cleanup_timeout_sec", 12.0)
        )
        self._ros_graph_best_effort_wait_sec = float(
            self._rt_cfg.get("ros_graph_best_effort_wait_sec", 1.0)
        )
        self._require_ros_graph_cleanup = bool(self._rt_cfg.get("require_ros_graph_cleanup", False))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def init(self, request: InitRequest) -> InitResponse:
        """
        - ROS node + spin thread only
        - Autoware itself is relaunched from reset() for each scenario
        """
        try:
            self._configure(request.output_dir, request.config)

            self._ensure_ros_node()
            self._stop_autoware_process()
            self._reset_adapter_state()
            self._set_map(request.map_name)
        except (InvalidAvRequest, AvPreconditionFailed, AvTimeout, AvUnavailable):
            raise

        logger.info("Autoware AV initialized. Autoware stack will be launched on reset.")
        return init_response(self._effective_config())

    def _effective_config(self) -> dict:
        """Return only normalized wrapper-specific settings that are actually in use."""
        return {
            "root": str(self._root),
            "ros_setup_script": str(self._ros_setup_script),
            "launch": {
                "package": str(self._launch_package),
                "file": str(self._launch_file),
                "headless": self._headless,
                "extra_args": [str(arg) for arg in self._extra_launch_args],
                "log": self._log_enabled,
                "log_path": self._autoware_log_path.name,
            },
            "data": {"data_path": str(self._data_path)},
            "vehicle": {
                "model": str(self._vehicle_model),
                "sensor_model": str(self._sensor_model),
            },
            "runtime": {
                "publish_agent_objects": self._publish_agent_objects,
                "timeout_sec": self._timeout_sec,
                "control_timeout_sec": self._control_timeout_sec,
                "wait_control": bool(self._rt_cfg.get("wait_control", False)),
                "allow_goal_modification": bool(self._rt_cfg.get("allow_goal_modification", False)),
                "engage_ready_stable_sec": self._engage_ready_stable_sec,
                "engage_retry_sec": self._engage_retry_sec,
                "engage_retry_interval_sec": self._engage_retry_interval_sec,
                "diagnostics_graph_enabled": self._diagnostics_graph_enabled,
                "diagnostics_as_precondition_failure": (self._diagnostics_as_precondition_failure),
                "precondition_diagnostic_hardware_ids": sorted(
                    str(value) for value in self._precondition_diagnostic_hardware_ids
                ),
                "lane_departure_boundary_check_enabled": (
                    self._lane_departure_boundary_check_enabled
                ),
                "runtime_param_timeout_sec": self._runtime_param_timeout_sec,
                "shutdown_interrupt_grace_sec": self._shutdown_interrupt_grace_sec,
                "shutdown_terminate_grace_sec": self._shutdown_terminate_grace_sec,
                "shutdown_kill_grace_sec": self._shutdown_kill_grace_sec,
                "process_group_cleanup_timeout_sec": self._process_group_cleanup_timeout_sec,
                "ros_graph_cleanup_timeout_sec": self._ros_graph_cleanup_timeout_sec,
                "ros_graph_best_effort_wait_sec": self._ros_graph_best_effort_wait_sec,
                "require_ros_graph_cleanup": self._require_ros_graph_cleanup,
            },
        }

    def reset(self, request: ResetRequest) -> ResetResponse:
        try:
            return self._reset(request)
        except (InvalidAvRequest, AvPreconditionFailed, AvTimeout, AvUnavailable):
            raise

    def _reset(self, request: ResetRequest) -> ResetResponse:
        """
        Reset AV internal state when simulator resets.

        1. Stop any previous Autoware process and wait for its ROS nodes to disappear
        2. Launch a fresh Autoware process
        3. Call InitializeLocalization service to set initial pose
        4. Call SetRoutePoints service to set route
        """
        output_related = request.output_dir
        sps = request.scenario_pack
        init_obs = request.initial_observation

        reset_started = time.monotonic()
        self._prepare_reset(output_related, sps, init_obs)

        logger.info("Relaunching Autoware for reset. map_path=%s", self._map_path)
        stage_started = time.monotonic()
        self._restart_autoware_stack()
        self._log_elapsed("reset.relaunch_autoware", stage_started)

        self._initialize_localization_for_reset()
        self._set_route_for_reset(sps)
        self._wait_for_planning_ready()
        self._engage_autoware()
        self._episode_ros_offset_ns = self._current_ros_time_ns
        self._log_elapsed("reset.total", reset_started)

        return ResetResponse(ctrl_cmd=self._prepare_control_payload())

    def _prepare_reset(
        self,
        output_related: str,
        sps: ScenarioPackData,
        init_obs: ObservationData,
    ) -> None:
        logger.debug("Setting timeout sec = %.2f", self._timeout_sec)
        self._output_dir = self._output_base / output_related
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._autoware_log_path = self._output_dir / self._autoware_log_path.name
        logger.debug("Output dir: %s", self._output_dir)

        self._ensure_ros_node()
        self._setup_sps(sps)

        self._reset_adapter_state()
        self._observation_normalizer.reset()
        try:
            ego, self._agents = self._observation_normalizer.normalize(init_obs, 0)
        except ObservationContractError as e:
            raise InvalidAvRequest(str(e)) from e
        self._kinematic = ego.kinematic
        self._last_sim_timestamp_ns = 0

    def _reset_adapter_state(self) -> None:
        self._initialized = False
        self._base_time_ns = self._current_ros_time_ns
        self._episode_ros_offset_ns = self._current_ros_time_ns
        self._sim_time_ns = 0
        self._latest_control = None
        self._latest_control_stamp = 0
        self._control_mode = autoware_vehicle_msgs.ControlModeReport.NO_COMMAND
        self._current_gear = autoware_vehicle_msgs.GearCommand.NONE
        self._quit_flag = False
        self._last_error = None
        self._kinematic = ObjectKinematicData()
        self._agents = []
        self._last_sim_timestamp_ns = None
        self._observation_normalizer.reset()
        self._reset_autoware_observed_state()

    def _initialize_localization_for_reset(self) -> None:
        logger.info("Initializing Autoware localization...")
        stage_started = time.monotonic()
        try:
            self._call_initialize_localization()
        except AvTimeout as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise
        except AvUnavailable:
            raise
        except InvalidAvRequest as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise
        self._log_elapsed("reset.initialize_localization.call", stage_started)

        logger.info("Waiting for Autoware localization...")
        stage_started = time.monotonic()
        self._wait_for_autoware_state(
            autoware_system_msgs.AutowareState.WAITING_FOR_ROUTE,
            "Autoware localization initialization timed out.",
        )
        self._log_elapsed("reset.initialize_localization.wait", stage_started)

    def _set_route_for_reset(self, sps: ScenarioPackData) -> None:
        logger.info("Setting Autoware route points...")
        stage_started = time.monotonic()
        try:
            time.sleep(1.0)
            self._call_set_route_points(sps)
        except AvTimeout as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise
        except AvUnavailable:
            raise
        except AvPreconditionFailed as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise
        self._log_elapsed("reset.set_route.call", stage_started)

        logger.info("Waiting for Autoware route...")
        stage_started = time.monotonic()
        self._wait_for_autoware_state_to_change(
            autoware_system_msgs.AutowareState.WAITING_FOR_ROUTE,
            "Autoware set route timed out.",
        )
        self._log_elapsed("reset.set_route.wait", stage_started)

    def _wait_for_planning_ready(self) -> None:
        logger.info("Waiting for Autoware planning...")
        stage_started = time.monotonic()
        self._wait_for_autoware_state(
            autoware_system_msgs.AutowareState.WAITING_FOR_ENGAGE,
            "Autoware planning timed out.",
        )
        self._log_elapsed("reset.planning.wait", stage_started)

        logger.info("Waiting for Autoware to be ready to engage...")
        stage_started = time.monotonic()
        self._wait_for_engage_ready_stable(
            "Autoware ready to engage timed out.",
        )
        self._log_elapsed("reset.ready_to_engage.wait", stage_started)

        logger.info("Autoware reset is planned and ready to engage.")

    def _wait_for_autoware_state(
        self,
        expected_state: int,
        timeout_message: str,
    ) -> None:
        self._wait_until(
            lambda: self._vehicle_state == expected_state,
            timeout_message,
            debug_message=lambda: (
                f"Waiting for Autoware state {expected_state}, current state: {self._vehicle_state}"
            ),
        )

    def _wait_for_autoware_state_to_change(
        self,
        current_state: int,
        timeout_message: str,
    ) -> None:
        self._wait_until(
            lambda: self._vehicle_state != current_state,
            timeout_message,
            debug_message=lambda: (
                f"Waiting for Autoware to leave state {current_state}, "
                f"current state: {self._vehicle_state}"
            ),
        )

    def _wait_until(
        self,
        predicate: Callable[[], bool],
        timeout_message: str,
        *,
        debug_message: str | Callable[[], str],
    ) -> None:
        start = time.time()
        while not predicate():
            logger.debug(debug_message() if callable(debug_message) else debug_message)
            if time.time() - start > self._timeout_sec:
                full_message = self._compose_timeout_message(timeout_message)
                precondition_message = self._diagnostics_precondition_message(full_message)
                if precondition_message is not None:
                    self._last_error = precondition_message
                    logger.error(precondition_message)
                    self._quit_flag = True
                    raise AvPreconditionFailed(precondition_message)

                self._last_error = full_message
                logger.error(full_message)
                self._quit_flag = True
                raise AvTimeout(full_message)
            time.sleep(0.1)

    def _is_ready_to_engage(self) -> bool:
        return (
            self._vehicle_state == autoware_system_msgs.AutowareState.WAITING_FOR_ENGAGE
            and self._operation_mode_state is not None
            and self._operation_mode_state.is_autonomous_mode_available
            and not self._operation_mode_state.is_in_transition
        )

    def _wait_for_engage_ready_stable(self, timeout_message: str) -> None:
        start = time.time()
        stable_since: float | None = None
        while True:
            if self._is_ready_to_engage():
                if stable_since is None:
                    stable_since = time.time()
                if time.time() - stable_since >= self._engage_ready_stable_sec:
                    return
            else:
                stable_since = None

            logger.debug(
                "Waiting for Autoware engage readiness to stabilize: state=%s, operation_mode=%s",
                self._vehicle_state,
                self._operation_mode_state,
            )
            if time.time() - start > self._timeout_sec:
                full_message = self._compose_timeout_message(timeout_message)
                precondition_message = self._diagnostics_precondition_message(full_message)
                if precondition_message is not None:
                    self._last_error = precondition_message
                    logger.error(precondition_message)
                    self._quit_flag = True
                    raise AvPreconditionFailed(precondition_message)

                self._last_error = full_message
                logger.error(full_message)
                self._quit_flag = True
                raise AvTimeout(full_message)
            time.sleep(0.1)

    def step(self, request: StepRequest) -> StepResponse:
        """
        Step function called at every sim step.
        """
        try:
            return self._step(request)
        except (InvalidAvRequest, AvPreconditionFailed, AvTimeout, AvUnavailable):
            raise

    def _step(self, request: StepRequest) -> StepResponse:
        obs = request.observation
        time_stamp_ns = request.timestamp_ns

        self._ensure_ros_node()
        if self._last_sim_timestamp_ns is None:
            raise InvalidAvRequest("Step requires a successful Reset")
        if time_stamp_ns < self._last_sim_timestamp_ns:
            raise InvalidAvRequest(
                f"Step timestamp {time_stamp_ns} precedes {self._last_sim_timestamp_ns}"
            )
        try:
            ego, self._agents = self._observation_normalizer.normalize(obs, time_stamp_ns)
        except ObservationContractError as e:
            raise InvalidAvRequest(str(e)) from e
        self._last_sim_timestamp_ns = time_stamp_ns
        self._sim_time_ns = time_stamp_ns
        self._current_ros_time_ns = self._episode_ros_offset_ns + self._sim_time_ns

        # Check if autoware is completed
        if self._vehicle_state == autoware_system_msgs.AutowareState.ARRIVED_GOAL:
            logger.info("Autoware has completed the route.")
            self._quit_flag = True
            return StepResponse(ctrl_cmd=ControlCommand(mode=ControlMode.NONE))

        # Check Autoware vehicle state
        if self._vehicle_state != autoware_system_msgs.AutowareState.DRIVING:
            logger.warning(f"Autoware not in driving mode, current state: {self._vehicle_state}")
            return StepResponse(ctrl_cmd=ControlCommand(mode=ControlMode.NONE))

        # Update ego's kinematic state
        self._kinematic = ego.kinematic

        # publish
        now = Time(nanoseconds=self._current_ros_time_ns)
        self._publish_manager.publish_all(now)

        if self._rt_cfg.get("wait_control", False):
            deadline = time.time() + self._control_timeout_sec
            while time.time() < deadline:
                if (
                    self._latest_control is not None
                    and self._latest_control.stamp.sec * 1e9 + self._latest_control.stamp.nanosec
                    > self._latest_control_stamp
                ):
                    break
                time.sleep(0.001)

        if self._latest_control is None:
            logger.warning("No control message received from Autoware, returning zero Ctrl")
            return StepResponse(ctrl_cmd=ControlCommand(mode=ControlMode.NONE))

        # Apply control
        self._latest_control_stamp = (
            self._latest_control.stamp.sec * 1e9 + self._latest_control.stamp.nanosec
        )

        return StepResponse(ctrl_cmd=self._prepare_control_payload())

    def stop(self) -> None:
        """Stop Autoware process + ROS node / executor"""
        self._stop_autoware_process()

        if self._executor and self._node:
            self._executor.remove_node(self._node)

        if self._node:
            self._node.destroy_node()
            self._node = None

        rclpy.try_shutdown()

        logger.info("Autoware AV stopped.")

    def should_quit(self) -> ShouldQuitResponse:
        """
        True if:
        - internal error / service failure happened (quit_flag set)
        - Autoware process exited unexpectedly
        """
        if self._quit_flag:
            msg = self._last_error or "Autoware AV quit flag set"
            logger.info("AutowareAV.should_quit: %s", msg)
            return ShouldQuitResponse(should_quit=True, msg=msg)

        # Autoware process 狀態
        if self._autoware_proc is not None and self._autoware_proc.poll() is not None:
            return_code = self._autoware_proc.returncode
            msg = f"Autoware process exited unexpectedly with return code {return_code}"
            logger.info(msg)
            return ShouldQuitResponse(should_quit=True, msg=msg)

        return ShouldQuitResponse(should_quit=False)

    # ------------------------------------------------------------------
    # ROS node / spin / process
    # ------------------------------------------------------------------
    def _ensure_ros_node(self) -> None:
        # check if rclpy is inited
        if not rclpy.ok():
            rclpy.init()

        if self._node is not None:
            return

        self._node = rclpy.create_node("autoware_av_adapter")
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._node.create_timer(1.0 / CLOCK_PUB_HZ, self._timer_callback)
        self._publish_manager = PublishManager()

        # QoS profile:
        # Reliability: RELIABLE
        # History (Depth): KEEP_LAST (1)
        # Durability: VOLATILE
        # Lifespan: Infinite
        # Deadline: Infinite
        # Liveliness: AUTOMATIC
        # Liveliness lease duration: Infinite

        qos_profile = QoSProfile(depth=1)

        # publishers
        self._kinematic_state_pub = self._node.create_publisher(
            nav_msgs.Odometry,
            "/localization/kinematic_state",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="kinematic_state",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_kinematic_state(t),
            )
        )

        self._accel_pub = self._node.create_publisher(
            geometry_msgs.AccelWithCovarianceStamped,
            "/localization/acceleration",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="accel",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_accel(t),
            )
        )

        self._objects_pub = self._node.create_publisher(
            autoware_perception_msgs.DetectedObjects,
            "/perception/object_recognition/detection/objects",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="dynamic_objects",
                rate_hz=10.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_dynamic_objects(t),
            )
        )

        self._dummy_pointcloud_pub = self._node.create_publisher(
            sensor_msgs.PointCloud2,
            "/perception/obstacle_segmentation/pointcloud",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._publish_manager.add(
            TopicPublisher(
                name="dummy_pointcloud",
                rate_hz=10.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_dummy_pointcloud(t),
            )
        )

        self._control_mode_pub = self._node.create_publisher(
            autoware_vehicle_msgs.ControlModeReport,
            "/vehicle/status/control_mode",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="control_mode",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_control_mode(t),
            )
        )

        self._gear_report_pub = self._node.create_publisher(
            autoware_vehicle_msgs.GearReport,
            "/vehicle/status/gear_status",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="gear_report",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_gear_report(t),
            )
        )

        self._steering_report_pub = self._node.create_publisher(
            autoware_vehicle_msgs.SteeringReport,
            "/vehicle/status/steering_status",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="steering_report",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_steering_report(t),
            )
        )

        self._velocity_report_pub = self._node.create_publisher(
            autoware_vehicle_msgs.VelocityReport,
            "/vehicle/status/velocity_status",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="velocity_report",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_velocity_report(t),
            )
        )

        self._occupancy_grid_pub = self._node.create_publisher(
            nav_msgs.OccupancyGrid,
            "/perception/occupancy_grid_map/map",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="occupancy_grid",
                rate_hz=10.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_occupancy_grid(t),
            )
        )

        self._clock_pub = self._node.create_publisher(
            rosgraph_msgs.Clock,
            "/clock",
            qos_profile,
        )
        self._publish_manager.add(
            TopicPublisher(
                name="clock",
                rate_hz=100.0,
                mode=PublishMode.ALWAYS,
                enabled=True,
                publish_fn=lambda t: self._publish_clock(t),
            )
        )

        self._tf_broadcaster = TransformBroadcaster(self._node)
        self._publish_manager.add(
            TopicPublisher(
                name="tf",
                rate_hz=40.0,
                mode=PublishMode.FIXED_RATE,
                enabled=True,
                publish_fn=lambda t: self._publish_tf(t),
            )
        )

        # subscribers
        self._control_sub = self._node.create_subscription(
            autoware_control_msgs.Control,
            "/control/command/control_cmd",
            # self._on_control,
            lambda msg: setattr(self, "_latest_control", msg),
            qos_profile,
        )

        self._autoware_state_sub = self._node.create_subscription(
            autoware_system_msgs.AutowareState,
            "/autoware/state",
            lambda msg: setattr(self, "_vehicle_state", msg.state),
            qos_profile,
        )

        if self._diagnostics_graph_enabled:
            self._diagnostics_graph_sub = self._node.create_subscription(
                tier4_system_msgs.DiagGraphStatus,
                "/diagnostics_graph/status",
                self._on_diagnostics_graph,
                QoSProfile(depth=1),
            )

        self._gear_cmd_sub = self._node.create_subscription(
            autoware_vehicle_msgs.GearCommand,
            "/control/command/gear_cmd",
            lambda msg: setattr(self, "_current_gear", msg.command),
            qos_profile,
        )

        self._autoware_motion_state_sub = self._node.create_subscription(
            autoware_adapi_v1_msgs.MotionState,
            "/api/motion/state",
            lambda msg: setattr(self, "_autoware_motion_state", msg.state),
            qos_profile,
        )

        self._operation_mode_state_sub = self._node.create_subscription(
            autoware_adapi_v1_msgs.OperationModeState,
            "/api/operation_mode/state",
            lambda msg: setattr(self, "_operation_mode_state", msg),
            qos_profile,
        )

        self._route_state_sub = self._node.create_subscription(
            autoware_adapi_v1_msgs.RouteState,
            "/api/routing/state",
            lambda msg: setattr(self, "_route_state", msg.state),
            qos_profile,
        )

        # services
        self._client_initial_localization = self._node.create_client(
            autoware_adapi_v1_msgs_srv.InitializeLocalization,
            "/api/localization/initialize",
        )
        self._client_clear_route = self._node.create_client(
            autoware_adapi_v1_msgs_srv.ClearRoute,
            "/api/routing/clear_route",
        )
        self._client_set_route_points = self._node.create_client(
            autoware_adapi_v1_msgs_srv.SetRoutePoints,
            "/api/routing/set_route_points",
        )
        self._client_change_to_stop = self._node.create_client(
            autoware_adapi_v1_msgs_srv.ChangeOperationMode,
            "/api/operation_mode/change_to_stop",
        )
        self._client_change_to_auto = self._node.create_client(
            autoware_adapi_v1_msgs_srv.ChangeOperationMode,
            "/api/operation_mode/change_to_autonomous",
        )

        # spin thread
        self._spin_thread = threading.Thread(target=self._spin, daemon=False)
        self._spin_thread.start()

    def _spin(self) -> None:
        assert self._executor is not None
        try:
            self._executor.spin()
        except Exception as e:  # noqa: BLE001
            logger.error(f"AutowareAV executor error: {e}")
            self._last_error = str(e)
            self._quit_flag = True
            # break

    def _on_diagnostics_graph(self, msg: tier4_system_msgs.DiagGraphStatus) -> None:
        try:
            active = []
            for status in msg.diags:
                level = self._diagnostic_level_to_int(status.level)
                if level == 0:
                    continue
                active.append(
                    (
                        level,
                        getattr(status, "name", ""),
                        status.hardware_id,
                        status.message,
                    )
                )
            self._latest_diagnostics = active
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to update diagnostics graph summary: %s", e)

    def _diagnostics_summary(self) -> str:
        if not self._latest_diagnostics:
            return ""

        level_names = {
            1: "WARN",
            2: "ERROR",
            3: "STALE",
        }
        parts = []
        for level, name, hardware_id, message in self._latest_diagnostics[:8]:
            label = level_names.get(level, str(level))
            source = name or hardware_id or "unknown"
            if message:
                parts.append(f"{label} {source}: {message}")
            else:
                parts.append(f"{label} {source}")
        if len(self._latest_diagnostics) > len(parts):
            parts.append(f"... {len(self._latest_diagnostics) - len(parts)} more")
        return "; ".join(parts)

    def _compose_timeout_message(self, timeout_message: str) -> str:
        diagnostic_summary = self._diagnostics_summary()
        if diagnostic_summary:
            return f"{timeout_message} Active diagnostics: {diagnostic_summary}"
        return timeout_message

    def _diagnostics_precondition_message(self, message: str) -> str | None:
        if not self._diagnostics_as_precondition_failure:
            return None

        for level, name, hardware_id, diagnostic_message in self._latest_diagnostics:
            source = hardware_id or name
            if source not in self._precondition_diagnostic_hardware_ids:
                continue
            if level < 2:
                continue
            detail = f"{source}: {diagnostic_message}" if diagnostic_message else source
            return f"Autoware precondition failed: {detail}. {message}"

        return None

    @staticmethod
    def _diagnostic_level_to_int(level: int | bytes | str) -> int:
        if isinstance(level, bytes):
            return level[0] if level else 0
        if isinstance(level, str):
            return ord(level[0]) if level else 0
        return int(level)

    def _log_elapsed(self, label: str, started: float) -> None:
        logger.debug("%s took %.3fs", label, time.monotonic() - started)

    def _restart_autoware_stack(self) -> None:
        logger.info("Stopping previous Autoware stack...")
        stage_started = time.monotonic()
        self._stop_autoware_process()
        self._log_elapsed("reset.stop_previous_autoware", stage_started)

        self._reset_autoware_observed_state()

        logger.info("Launching new Autoware stack...")
        stage_started = time.monotonic()
        self._launch_autoware()
        self._log_elapsed("reset.launch_process", stage_started)

        logger.info("Waiting for Autoware services...")
        stage_started = time.monotonic()
        self._wait_for_service(self._client_initial_localization, "InitializeLocalization")
        self._wait_for_service(self._client_set_route_points, "SetRoutePoints")
        self._wait_for_service(self._client_change_to_auto, "ChangeOperationMode")
        logger.info("Autoware services are ready.")
        self._log_elapsed("reset.wait_services", stage_started)

        self._apply_runtime_autoware_params()

    def _launch_autoware(self) -> None:
        if self._map_path is None:
            raise RuntimeError("Cannot launch Autoware before map_path is configured")

        if self._autoware_proc is not None and self._autoware_proc.poll() is None:
            raise RuntimeError("Autoware process is already running")

        launch_parts = [
            f"cd {shlex.quote(str(self._root))}",
            f"source {shlex.quote(str(self._ros_setup_script))}",
            f"""ros2 launch {shlex.quote(str(self._launch_package))} {shlex.quote(str(self._launch_file))} \
            map_path:={shlex.quote(str(self._map_path.parent))} \
            lanelet2_map_file:={shlex.quote(self._map_path.name)} \
            data_path:={shlex.quote(str(self._data_path))} \
            vehicle_model:={shlex.quote(str(self._vehicle_model))} \
            sensor_model:={shlex.quote(str(self._sensor_model))} \
            launch_sensing:=false \
            launch_localization:=false \
            launch_perception:=false \
            launch_vehicle_interface:=false \
            system_run_mode:=planning_simulation \
            launch_system_monitor:=false \
            launch_dummy_diag_publisher:=true \
            enable_all_modules_auto_mode:=true \
            is_simulation:=true \
            rviz:={"false" if self._headless else "true"} \
            """,
        ]
        launch_parts.extend(self._extra_launch_args)

        full_cmd = " && ".join(launch_parts)
        logger.debug("Launching Autoware: %s", full_cmd)
        # subprocess.Popen dups the file descriptor into the child, so
        # closing the parent handle here doesn't disrupt the long-running
        # Autoware process — `with` is safe and avoids the leak.
        if self._log_enabled:
            with open(self._autoware_log_path, "wb", buffering=0) as log:
                self._autoware_proc = subprocess.Popen(
                    ["bash", "-lc", full_cmd],
                    stdout=log,
                    stderr=log,
                    preexec_fn=os.setsid,
                )
        else:
            self._autoware_proc = subprocess.Popen(
                ["bash", "-lc", full_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

    def _apply_runtime_autoware_params(self) -> None:
        if self._lane_departure_boundary_check_enabled:
            return

        logger.warning(
            "Disabling Autoware lane departure boundary check via runtime config. "
            "Autoware may engage from poses that it would normally reject as unsafe."
        )
        self._set_ros_param(
            "/control/trajectory_follower/lane_departure_checker_node",
            "boundary_departure_checker",
            "false",
        )

    def _set_ros_param(self, node_name: str, param_name: str, value: str) -> None:
        cmd = (
            f"source {shlex.quote(str(self._ros_setup_script))} && "
            f"ros2 param set {shlex.quote(node_name)} "
            f"{shlex.quote(param_name)} {shlex.quote(value)}"
        )
        try:
            completed = subprocess.run(
                ["bash", "-lc", cmd],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._runtime_param_timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            msg = (
                f"Timed out while setting Autoware parameter {node_name}.{param_name} "
                f"after {self._runtime_param_timeout_sec}s"
            )
            self._last_error = msg
            self._quit_flag = True
            raise AvTimeout(msg) from e

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            msg = f"Failed to set Autoware parameter {node_name}.{param_name}: {detail}"
            self._last_error = msg
            self._quit_flag = True
            raise AvUnavailable(msg)

    def _stop_autoware_process(self) -> None:
        """
        Stop Autoware process group quickly, with escalating signals if needed.
        """
        previous_nodes = self._list_autoware_ros_nodes()
        logger.debug("Found %d previous Autoware ROS nodes before shutdown.", len(previous_nodes))

        if self._autoware_proc is None:
            stage_started = time.monotonic()
            self._wait_for_autoware_ros_nodes_gone(previous_nodes, required=True)
            self._log_elapsed("reset.wait_ros_graph_cleanup", stage_started)
            return

        pgid: int | None = None
        with suppress(ProcessLookupError):
            pgid = os.getpgid(self._autoware_proc.pid)

        if self._autoware_proc.poll() is not None:
            self._autoware_proc = None
            self._wait_for_process_group_gone(pgid)
            stage_started = time.monotonic()
            self._wait_for_autoware_ros_nodes_gone(
                previous_nodes,
                required=self._require_ros_graph_cleanup,
            )
            self._log_elapsed("reset.wait_ros_graph_cleanup", stage_started)
            return

        logger.debug("Stopping Autoware process group...")

        try:
            if pgid is None:
                pgid = os.getpgid(self._autoware_proc.pid)
            self._signal_autoware_process_group(
                pgid,
                signal.SIGINT,
                self._shutdown_interrupt_grace_sec,
                "SIGINT",
            )
            if self._autoware_proc.poll() is None:
                self._signal_autoware_process_group(
                    pgid,
                    signal.SIGTERM,
                    self._shutdown_terminate_grace_sec,
                    "SIGTERM",
                )
            if self._autoware_proc.poll() is None:
                self._signal_autoware_process_group(
                    pgid,
                    signal.SIGKILL,
                    self._shutdown_kill_grace_sec,
                    "SIGKILL",
                )

        except ProcessLookupError:
            pass

        finally:
            self._autoware_proc = None
            self._wait_for_process_group_gone(pgid)
            stage_started = time.monotonic()
            self._wait_for_autoware_ros_nodes_gone(
                previous_nodes,
                required=self._require_ros_graph_cleanup,
            )
            self._log_elapsed("reset.wait_ros_graph_cleanup", stage_started)

    def _signal_autoware_process_group(
        self,
        pgid: int,
        sig: signal.Signals,
        wait_sec: float,
        label: str,
    ) -> None:
        assert self._autoware_proc is not None

        logger.debug("Sending %s to Autoware process group %s", label, pgid)
        with suppress(ProcessLookupError):
            os.killpg(pgid, sig)

        try:
            self._autoware_proc.wait(timeout=wait_sec)
            logger.debug("Autoware process exited after %s", label)
        except subprocess.TimeoutExpired as e:
            if sig == signal.SIGKILL:
                msg = f"Autoware process did not exit after {label}"
                self._last_error = msg
                self._quit_flag = True
                raise AvTimeout(msg) from e
            logger.debug("Autoware still running after %s grace %.2fs", label, wait_sec)

    def _wait_for_process_group_gone(self, pgid: int | None) -> None:
        if pgid is None:
            return

        if self._wait_for_process_group_exit(pgid, self._process_group_cleanup_timeout_sec):
            return

        members = self._process_group_members(pgid)
        logger.warning(
            "Autoware process group %s still has %d members after %.1fs cleanup: %s",
            pgid,
            len(members),
            self._process_group_cleanup_timeout_sec,
            members,
        )
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)

        if self._wait_for_process_group_exit(pgid, self._shutdown_kill_grace_sec):
            return

        msg = (
            f"Autoware process group {pgid} still exists "
            f"after SIGKILL shutdown cleanup: {self._process_group_members(pgid)}"
        )
        self._last_error = msg
        self._quit_flag = True
        raise AvTimeout(msg)

    def _wait_for_process_group_exit(self, pgid: int, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                logger.debug("Autoware process group %s is gone.", pgid)
                return True
            time.sleep(0.1)
        return False

    def _process_group_members(self, pgid: int) -> list[str]:
        members: list[str] = []
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text(encoding="utf-8")
                fields = stat.rsplit(")", maxsplit=1)[1].split()
                proc_pgid = int(fields[2])
                if proc_pgid != pgid:
                    continue

                pid = stat_path.parent.name
                comm = stat.split("(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0]
                state = fields[0]
                members.append(f"{pid}:{comm}:{state}")
            except (OSError, IndexError, ValueError):
                continue
        return members

    def _list_autoware_ros_nodes(self) -> set[str]:
        if self._node is None:
            return set()

        nodes: set[str] = set()
        for name, namespace in self._node.get_node_names_and_namespaces():
            namespace = namespace.rstrip("/")
            full_name = f"{namespace}/{name}" if namespace else f"/{name}"
            if full_name.startswith(AUTOWARE_NODE_PREFIXES) or full_name in AUTOWARE_NODE_NAMES:
                nodes.add(full_name)
        return nodes

    def _wait_for_autoware_ros_nodes_gone(
        self,
        previous_nodes: set[str],
        *,
        required: bool,
    ) -> None:
        if not previous_nodes:
            return

        timeout_sec = (
            self._ros_graph_cleanup_timeout_sec
            if required
            else self._ros_graph_best_effort_wait_sec
        )
        deadline = time.time() + timeout_sec
        next_log_time = 0.0
        while time.time() < deadline:
            remaining = previous_nodes & self._list_autoware_ros_nodes()
            if not remaining:
                logger.debug("Previous Autoware ROS nodes have been removed from the graph.")
                return
            now = time.time()
            if now >= next_log_time:
                logger.debug(
                    "Waiting for %d previous Autoware ROS nodes to leave graph: %s",
                    len(remaining),
                    sorted(remaining),
                )
                next_log_time = now + 1.0
            time.sleep(0.2)

        remaining = previous_nodes & self._list_autoware_ros_nodes()
        if not required:
            logger.warning(
                "Continuing after %.1fs with %d stale Autoware ROS graph nodes still visible: %s",
                timeout_sec,
                len(remaining),
                sorted(remaining),
            )
            return

        msg = (
            "Previous Autoware ROS nodes still visible "
            f"after {timeout_sec}s shutdown cleanup: {sorted(remaining)}"
        )
        self._last_error = msg
        self._quit_flag = True
        raise AvTimeout(msg)

    def _reset_autoware_observed_state(self) -> None:
        self._vehicle_state = None
        self._operation_mode_state = None
        self._autoware_motion_state = None
        self._route_state = None
        self._latest_control = None
        self._latest_control_stamp = 0
        self._latest_diagnostics = []

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _timer_callback(self):
        if not self._initialized:
            self._base_time_ns += int((1.0 / CLOCK_PUB_HZ) * 1e9)
            self._current_ros_time_ns = self._base_time_ns

            now = Time(nanoseconds=self._current_ros_time_ns)
            self._publish_manager.publish_all(now)

    # ------------------------------------------------------------------
    # AD API calls
    # ------------------------------------------------------------------
    def _wait_for_service(self, client, name: str, timeout_sec: float | None = None) -> None:
        timeout = timeout_sec or self._timeout_sec
        start = time.time()
        while not client.wait_for_service(timeout_sec=1.0):
            if time.time() - start > timeout:
                msg = f"Service {name} not available after {timeout}s"
                self._last_error = msg
                self._quit_flag = True
                raise AvTimeout(msg)
            logger.debug(f"Waiting for Autoware service {name}...")
        logger.debug("Service %s is available.", name)

    def _call_initialize_localization(self) -> None:
        assert self._node is not None
        now = Time(nanoseconds=self._current_ros_time_ns).to_msg()

        t = geometry_msgs.TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(self._kinematic.x)
        t.transform.translation.y = float(self._kinematic.y)
        t.transform.translation.z = float(self._kinematic.z)

        # Euler angle to quaternion
        ego_yaw = self._kinematic.yaw
        qz, qw = self._yaw_to_quat(ego_yaw)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        req = autoware_adapi_v1_msgs_srv.InitializeLocalization.Request()
        pose_msg = geometry_msgs.PoseWithCovarianceStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = float(self._kinematic.x)
        pose_msg.pose.pose.position.y = float(self._kinematic.y)
        pose_msg.pose.pose.position.z = float(self._kinematic.z)
        logger.debug(
            "Setting initial position: x=%s, y=%s, z=%s, h_raw=%s, h_ros=%s, speed=%s",
            self._kinematic.x,
            self._kinematic.y,
            self._kinematic.z,
            self._kinematic.yaw,
            ego_yaw,
            self._kinematic.speed,
        )
        qz, qw = self._yaw_to_quat(ego_yaw)
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw

        pose_msg.pose.covariance = [0.0] * 36
        sigma_pos = 1e-3  # 1 mm
        sigma_ang = 1e-4  # 0.0001 rad ~ 0.0057 deg

        pose_msg.pose.covariance[0] = sigma_pos**2  # x
        pose_msg.pose.covariance[7] = sigma_pos**2  # y
        pose_msg.pose.covariance[14] = sigma_pos**2  # z

        pose_msg.pose.covariance[21] = sigma_ang**2  # roll
        pose_msg.pose.covariance[28] = sigma_ang**2  # pitch
        pose_msg.pose.covariance[35] = sigma_ang**2  # yaw

        req.pose = [pose_msg]

        fut = self._client_initial_localization.call_async(req)

        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < self._timeout_sec:
            time.sleep(0.01)
        if not fut.done():
            raise AvTimeout(f"InitializeLocalization response timeout after {self._timeout_sec}s")

        res = fut.result()
        if res is None or not res.status.success:
            status_msg = getattr(res.status, "message", None) if res else "no response"
            code = getattr(res.status, "code", "unknown") if res else "no response"
            succ = getattr(res.status, "success", "unknown") if res else "no response"
            msg = (
                f"InitializeLocalization failed: code={code}, success={succ}, message={status_msg}"
            )
            raise AvUnavailable(msg)

        logger.debug("Called InitializeLocalization service.")

    def _call_set_route_points(self, sps: ScenarioPackData) -> None:
        assert self._node is not None
        req = autoware_adapi_v1_msgs_srv.SetRoutePoints.Request()
        req.header.frame_id = "map"
        req.header.stamp = Time(nanoseconds=self._current_ros_time_ns).to_msg()

        gp = sps.ego.goal_config.position
        goal = geometry_msgs.Pose()
        goal.position.x = float(gp.world.x)
        goal.position.y = float(gp.world.y)
        goal.position.z = float(gp.world.z)

        qz, qw = self._yaw_to_quat(gp.world.h)
        goal.orientation.z = qz
        goal.orientation.w = qw

        req.goal = goal
        req.waypoints = []

        req.option.allow_goal_modification = bool(
            self._rt_cfg.get("allow_goal_modification", False)
        )

        fut = self._client_set_route_points.call_async(req)
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < self._timeout_sec:
            time.sleep(0.01)
        if not fut.done():
            raise AvTimeout(f"SetRoutePoints response timeout after {self._timeout_sec}s")
        res = fut.result()
        if res is None or not res.status.success:
            status_msg = getattr(res.status, "message", None) if res else "no response"
            msg = f"SetRoutePoints failed: message={status_msg}, start: ({self._kinematic.x}, {self._kinematic.y}, {self._kinematic.z}, {self._kinematic.yaw}), destination: ({goal.position.x}, {goal.position.y}, {goal.position.z}, {gp.world.h})"
            raise AvPreconditionFailed(msg)

    def _call_clear_route(self) -> None:
        # Clear route
        req = autoware_adapi_v1_msgs_srv.ClearRoute.Request()
        fut = self._client_clear_route.call_async(req)
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < self._timeout_sec:
            time.sleep(0.01)
        if not fut.done():
            raise AvTimeout(f"ClearRoute response timeout after {self._timeout_sec}s")
        res = fut.result()
        if res is None or not res.status.success:
            status_msg = getattr(res.status, "message", None) if res else "no response"
            code = getattr(res.status, "code", "unknown") if res else "no response"
            succ = getattr(res.status, "success", "unknown") if res else "no response"
            msg = f"ClearRoute failed: code={code}, success={succ}, message={status_msg}"
            raise AvUnavailable(msg)

    def _call_change_to_stop(self) -> None:
        assert self._node is not None
        req = autoware_adapi_v1_msgs_srv.ChangeOperationMode.Request()
        fut = self._client_change_to_stop.call_async(req)
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < self._timeout_sec:
            time.sleep(0.01)
        if not fut.done():
            raise AvTimeout(
                f"ChangeOperationMode(STOP) response timeout after {self._timeout_sec}s"
            )
        res = fut.result()
        if res is None or not res.status.success:
            msg = f"ChangeOperationMode(STOP) failed: {getattr(res.status, 'message', 'unknown') if res else 'no response'}"
            self._last_error = msg
            logger.error(msg)
            raise AvUnavailable(msg)

    def _engage_autoware(self) -> None:
        logger.info("Engaging Autoware autonomous mode...")
        stage_started = time.monotonic()
        try:
            self._call_change_to_autonomous_with_retry()
            self._control_mode = autoware_vehicle_msgs.ControlModeReport.AUTONOMOUS
        except AvTimeout as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise
        except AvUnavailable as e:
            self._quit_flag = True
            self._last_error = str(e)
            raise AvUnavailable("Failed to change Autoware to autonomous mode.") from e

        self._wait_for_autoware_state(
            autoware_system_msgs.AutowareState.DRIVING,
            "Autoware change to autonomous mode timed out.",
        )
        self._initialized = True
        logger.info("Autoware is running.")
        self._log_elapsed("reset.engage", stage_started)

    def _call_change_to_autonomous_with_retry(self) -> None:
        deadline = time.time() + self._engage_retry_sec

        while True:
            try:
                self._wait_for_engage_ready_stable(
                    "Autoware ready to engage timed out while retrying autonomous mode.",
                )
                self._call_change_to_autonomous()
                return
            except AvUnavailable as e:
                if time.time() >= deadline:
                    raise
                logger.warning(
                    "Autoware autonomous mode change was rejected; retrying for %.1fs: %s",
                    max(0.0, deadline - time.time()),
                    e,
                )
                time.sleep(self._engage_retry_interval_sec)

    def _call_change_to_autonomous(self) -> None:
        assert self._node is not None

        req = autoware_adapi_v1_msgs_srv.ChangeOperationMode.Request()
        fut = self._client_change_to_auto.call_async(req)
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < self._timeout_sec:
            time.sleep(0.01)
        if not fut.done():
            raise AvTimeout(
                f"ChangeOperationMode(AUTONOMOUS) response timeout after {self._timeout_sec}s"
            )
        res = fut.result()
        if res is None or not res.status.success:
            msg = f"ChangeOperationMode(AUTONOMOUS) failed: {getattr(res.status, 'message', 'unknown') if res else 'no response'}"
            self._last_error = msg
            self._quit_flag = True
            logger.error(msg)
            raise AvUnavailable(msg)

    # ------------------------------------------------------------------
    # publish helpers
    # ------------------------------------------------------------------
    def _publish_kinematic_state(self, t: rclpy.time.Time) -> None:
        assert self._node is not None

        now = t.to_msg()

        # ODOMETRY
        msg = nav_msgs.Odometry()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = self._kinematic.x
        msg.pose.pose.position.y = self._kinematic.y
        msg.pose.pose.position.z = self._kinematic.z

        yaw = self._kinematic.yaw
        qz, qw = self._yaw_to_quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = self._kinematic.speed
        msg.twist.twist.angular.z = self._kinematic.yaw_rate

        self._kinematic_state_pub.publish(msg)

    def _publish_accel(self, t: rclpy.time.Time) -> None:
        now = t.to_msg()
        accel = geometry_msgs.AccelWithCovarianceStamped()
        accel.header.stamp = now
        accel.header.frame_id = "base_link"
        accel.accel.accel.linear.x = self._kinematic.acceleration
        accel.accel.accel.angular.z = self._kinematic.yaw_acceleration
        self._accel_pub.publish(accel)

    def _publish_dynamic_objects(self, t: rclpy.time.Time) -> None:
        msg = autoware_perception_msgs.DetectedObjects()
        msg.header.stamp = t.to_msg()

        msg.header.frame_id = "map"

        if not self._publish_agent_objects:
            self._objects_pub.publish(msg)
            return

        for ag in self._agents:
            obj = autoware_perception_msgs.DetectedObject()

            # 1. existence_probability
            obj.existence_probability = 1.0

            # 2. Classification
            clas = autoware_perception_msgs.ObjectClassification()
            clas.label = OBJECT_TYPE_MAP.get(
                ag.type, autoware_perception_msgs.ObjectClassification.UNKNOWN
            )
            clas.probability = 1.0
            obj.classification = [clas]

            # 3. Shape
            shp = autoware_perception_msgs.Shape()

            if ag.shape is None:
                logger.warning("Skipping dynamic object without shape")
                continue

            if ag.shape.type == ShapeType.CYLINDER:
                shp.type = autoware_perception_msgs.Shape.CYLINDER
            elif ag.shape.type == ShapeType.BOUNDING_BOX:
                shp.type = autoware_perception_msgs.Shape.BOUNDING_BOX
            elif ag.shape.type == ShapeType.POLYGON:
                shp.type = autoware_perception_msgs.Shape.POLYGON
            else:
                raise InvalidAvRequest(f"Unknown shape type: {ag.shape.type}")

            if ag.shape.type != ShapeType.POLYGON:
                shp.dimensions.x = ag.shape.dimensions.x
                shp.dimensions.y = ag.shape.dimensions.y
                shp.dimensions.z = ag.shape.dimensions.z
            else:
                for pt in ag.shape.vertices:
                    p = geometry_msgs.Point32()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z
                    shp.footprint.points.append(p)
                shp.height = ag.shape.dimensions.z

            obj.shape = shp

            # 4. Kinematics
            kin = autoware_perception_msgs.DetectedObjectKinematics()

            kin.orientation_availability = 2  # (0:UNAVAILABLE, 1:SIGN_UNKNOWN, 2:AVAILABLE)
            kin.has_position_covariance = True

            # Pose
            x, y, z, qx, qy, qz, qw = compose_shape_center_pose(
                ag.kinematic,
                ag.shape.center,
            )
            kin.pose_with_covariance.pose.position.x = x
            kin.pose_with_covariance.pose.position.y = y
            kin.pose_with_covariance.pose.position.z = z

            kin.pose_with_covariance.pose.orientation.x = qx
            kin.pose_with_covariance.pose.orientation.y = qy
            kin.pose_with_covariance.pose.orientation.z = qz
            kin.pose_with_covariance.pose.orientation.w = qw

            sigma_pos = 1e-3  # 1 mm
            sigma_ang = 1e-4  # 0.0001 rad ~ 0.0057 deg

            kin.pose_with_covariance.covariance[0] = sigma_pos**2
            kin.pose_with_covariance.covariance[7] = sigma_pos**2
            kin.pose_with_covariance.covariance[14] = sigma_pos**2
            kin.pose_with_covariance.covariance[21] = sigma_ang**2
            kin.pose_with_covariance.covariance[28] = sigma_ang**2
            kin.pose_with_covariance.covariance[35] = sigma_ang**2

            # Twist
            kin.has_twist = True
            kin.twist_with_covariance.twist.linear.x = ag.kinematic.speed
            kin.twist_with_covariance.twist.angular.z = ag.kinematic.yaw_rate

            sigma_v = 0.02  # m/s
            sigma_w = 0.01  # rad/s

            kin.has_twist_covariance = True
            kin.twist_with_covariance.covariance[0] = sigma_v**2
            kin.twist_with_covariance.covariance[7] = sigma_v**2
            kin.twist_with_covariance.covariance[14] = sigma_v**2
            kin.twist_with_covariance.covariance[21] = sigma_w**2
            kin.twist_with_covariance.covariance[28] = sigma_w**2
            kin.twist_with_covariance.covariance[35] = sigma_w**2

            obj.kinematics = kin
            msg.objects.append(obj)
        self._objects_pub.publish(msg)

    def _publish_dummy_pointcloud(self, t: rclpy.time.Time) -> None:
        # Empty PointCloud2
        msg = sensor_msgs.PointCloud2()
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = "base_link"
        msg.height = 1
        msg.width = 0
        msg.is_dense = True
        msg.is_bigendian = False
        x = sensor_msgs.PointField()
        x.name = "x"
        x.offset = 0
        x.datatype = sensor_msgs.PointField.FLOAT32
        x.count = 1
        y = sensor_msgs.PointField()
        y.name = "y"
        y.offset = 4
        y.datatype = sensor_msgs.PointField.FLOAT32
        y.count = 1
        z = sensor_msgs.PointField()
        z.name = "z"
        z.offset = 8
        z.datatype = sensor_msgs.PointField.FLOAT32
        z.count = 1
        intensity = sensor_msgs.PointField()
        intensity.name = "intensity"
        intensity.offset = 12
        intensity.datatype = sensor_msgs.PointField.UINT8
        intensity.count = 1
        returntype = sensor_msgs.PointField()
        returntype.name = "return_type"
        returntype.offset = 13
        returntype.datatype = sensor_msgs.PointField.UINT8
        returntype.count = 1
        channel = sensor_msgs.PointField()
        channel.name = "channel"
        channel.offset = 14
        channel.datatype = sensor_msgs.PointField.UINT16
        channel.count = 1
        msg.fields = [x, y, z, intensity, returntype, channel]
        msg.point_step = 16
        msg.row_step = 0
        msg.data = b""
        self._dummy_pointcloud_pub.publish(msg)

    def _publish_tf(self, t: rclpy.time.Time) -> None:
        if self._kinematic is None:
            logger.warning("No kinematic state skipping TF publish")
            return
        now = t.to_msg()

        t = geometry_msgs.TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = self._kinematic.x
        t.transform.translation.y = self._kinematic.y
        t.transform.translation.z = self._kinematic.z

        t.transform.rotation.z, t.transform.rotation.w = self._yaw_to_quat(self._kinematic.yaw)

        # publish
        self._tf_broadcaster.sendTransform(t)

    def _publish_control_mode(self, t: rclpy.time.Time) -> None:
        msg = autoware_vehicle_msgs.ControlModeReport()
        msg.stamp = t.to_msg()
        msg.mode = self._control_mode
        self._control_mode_pub.publish(msg)

    def _publish_gear_report(self, t: rclpy.time.Time) -> None:
        msg = autoware_vehicle_msgs.GearReport()
        msg.stamp = t.to_msg()
        msg.report = self._current_gear
        self._gear_report_pub.publish(msg)

    def _publish_steering_report(self, t: rclpy.time.Time) -> None:
        msg = autoware_vehicle_msgs.SteeringReport()
        msg.stamp = t.to_msg()
        # TODO: This should be obtained from vehicle state
        angle = self._latest_control.lateral.steering_tire_angle if self._latest_control else 0.0
        msg.steering_tire_angle = angle
        self._steering_report_pub.publish(msg)

    def _publish_velocity_report(self, t: rclpy.time.Time) -> None:
        msg = autoware_vehicle_msgs.VelocityReport()
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = "base_link"
        msg.longitudinal_velocity = self._kinematic.speed
        msg.lateral_velocity = 0.0
        msg.heading_rate = self._kinematic.yaw_rate

        self._velocity_report_pub.publish(msg)

    def _publish_occupancy_grid(self, t: rclpy.time.Time) -> None:
        # TODO: Check how to get actual occupancy grid from Autoware
        msg = nav_msgs.OccupancyGrid()
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = "map"

        # Empty map info
        msg.info.resolution = 0.5  # size of each grid cell (m)
        msg.info.width = 200  # number of grids (width)
        msg.info.height = 200  # number of grids (height)
        msg.info.origin.position.x = -50.0  # bottom-left x (m)
        msg.info.origin.position.y = -50.0  # bottom-left y (m)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Unknown occupancy
        msg.data = [-1] * (msg.info.width * msg.info.height)

        self._occupancy_grid_pub.publish(msg)

    def _publish_clock(self, t: rclpy.time.Time) -> None:
        msg = rosgraph_msgs.Clock()
        msg.clock = t.to_msg()
        self._clock_pub.publish(msg)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_control_payload(self) -> ControlCommand:
        if self._latest_control is None:
            logger.warning("No control message received from Autoware, returning zero Ctrl")
            return ControlCommand(mode=ControlMode.NONE)

        steer = float(self._latest_control.lateral.steering_tire_angle)
        if bool(self._latest_control.lateral.is_defined_steering_tire_rotation_rate):
            steer_speed = float(self._latest_control.lateral.steering_tire_rotation_rate)
        else:
            steer_speed = 0.0

        speed = clamp_ackermann_speed(float(self._latest_control.longitudinal.velocity))

        acceleration = None
        if bool(self._latest_control.longitudinal.is_defined_acceleration):
            acceleration = float(self._latest_control.longitudinal.acceleration)

        jerk = None
        if bool(self._latest_control.longitudinal.is_defined_jerk):
            jerk = float(self._latest_control.longitudinal.jerk)

        payload = {
            "steer": steer,
            "steer_speed": steer_speed,
            "speed": speed,
        }
        if acceleration is not None:
            payload["acceleration"] = acceleration
        if jerk is not None:
            payload["jerk"] = jerk

        try:
            validate_ackermann_payload(payload)
        except ValueError as e:
            self._last_error = str(e)
            self._quit_flag = True
            raise AvUnavailable(str(e)) from e

        return ControlCommand(mode=ControlMode.ACKERMANN, payload=payload)

    def _set_map(self, map_name: str) -> bool:
        map_full_path = Path(f"/mnt/map/osm/{map_name}.osm").resolve()
        if not map_full_path.exists():
            raise InvalidAvRequest(f"Autoware map file not found: {map_full_path}")
        is_changed = self._map_path != map_full_path
        self._map_path = map_full_path
        return is_changed

    def _setup_sps(self, sps: ScenarioPackData) -> bool:
        """
        Update map path from ScenarioPack.
        Return True if map path has changed.
        """

        # Update sps
        self._sps = sps
        return self._set_map(sps.map_name)

    def _stop_autoware_vehicle(self) -> None:
        try:
            self._call_change_to_stop()
        except Exception as e:
            logger.warning(f"Failed to change Autoware to STOP mode: {e}")
            # continue anyway

        # wait until Autoware reports stopped state or timeout
        start = time.time()
        while rclpy.ok() and (
            self._autoware_motion_state is None
            or self._autoware_motion_state != autoware_adapi_v1_msgs.MotionState.STOPPED
        ):
            logger.debug(
                f"Waiting for Autoware to stop... current state: {self._autoware_motion_state}, Elapsed time: {time.time() - start:.2f}s"
            )
            if time.time() - start > self._timeout_sec:
                logger.warning(
                    f"Timeout while waiting for Autoware to stop after {self._timeout_sec}s"
                )
                break
            time.sleep(0.1)

        try:
            self._call_clear_route()
        except Exception as e:
            logger.warning(f"Failed to clear Autoware route: {e}")
            # continue anyway

        # check self._route_state to be UNSET and autoware_state to be WAITING_FOR_ROUTE or WAITING_FOR_INITIAL_POSE
        start = time.time()
        while (
            self._route_state != autoware_adapi_v1_msgs.RouteState.UNSET
            or self._vehicle_state is None
            or self._vehicle_state >= autoware_system_msgs.AutowareState.PLANNING
        ):
            logger.debug(
                f"Waiting for Autoware to clear route... current route state: {self._route_state}, elapsed time: {time.time() - start:.2f}s"
            )
            if time.time() - start > self._timeout_sec:
                logger.warning(
                    f"Timeout while waiting for Autoware to clear route after {self._timeout_sec}s"
                )
                break
            time.sleep(0.1)

        logger.debug("Autoware vehicle stopped and route cleared.")
        logger.debug(
            "Current Autoware state: %s, motion state: %s, route state: %s",
            self._vehicle_state,
            self._autoware_motion_state,
            self._route_state,
        )

    @staticmethod
    def _yaw_to_quat(yaw: float) -> tuple[float, float]:
        """Supposing roll=pitch=0, return quaternion z,w from yaw"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return sy, cy

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Supposing roll=pitch=0, return yaw from quaternion"""
        return math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
