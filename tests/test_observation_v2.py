import math

import pytest
from pisa_api.av import (
    ObjectKinematicData,
    ObjectStateData,
    ObservationData,
    ObservedAgentData,
    ShapeCenterPoseData,
)

from autoware_wrapper.geometry import compose_shape_center_pose, states_from_observation


def _state(x: float) -> ObjectStateData:
    return ObjectStateData(kinematic=ObjectKinematicData(x=x))


def test_observation_uses_explicit_ego_and_agent_states() -> None:
    ego = _state(100.0)
    first = _state(1.0)
    second = _state(2.0)
    observation = ObservationData(
        ego=ego,
        agents=[
            ObservedAgentData(state=first),
            ObservedAgentData(state=second, tracking_id=42, entity_name="privileged-name"),
        ],
    )

    actual_ego, agents = states_from_observation(observation)

    assert actual_ego is ego
    assert agents == [first, second]


def test_agent_permutation_does_not_change_ego_or_detected_state_set() -> None:
    ego = _state(100.0)
    first = ObservedAgentData(state=_state(1.0), tracking_id=11)
    second = ObservedAgentData(state=_state(2.0))

    ego_a, agents_a = states_from_observation(ObservationData(ego=ego, agents=[first, second]))
    ego_b, agents_b = states_from_observation(ObservationData(ego=ego, agents=[second, first]))

    assert ego_a == ego_b == ego
    assert {agent.kinematic.x for agent in agents_a} == {
        agent.kinematic.x for agent in agents_b
    }


def test_new_disappearing_agents_follow_current_observation_only() -> None:
    ego = _state(100.0)
    _, initial = states_from_observation(
        ObservationData(ego=ego, agents=[ObservedAgentData(state=_state(1.0))])
    )
    _, replacement = states_from_observation(
        ObservationData(ego=ego, agents=[ObservedAgentData(state=_state(2.0))])
    )
    _, empty = states_from_observation(ObservationData(ego=ego))

    assert [state.kinematic.x for state in initial] == [1.0]
    assert [state.kinematic.x for state in replacement] == [2.0]
    assert empty == []


def test_shape_center_offset_and_orientation_are_composed() -> None:
    pose = compose_shape_center_pose(
        ObjectKinematicData(x=10.0, y=20.0, z=1.0, yaw=math.pi / 2),
        ShapeCenterPoseData(x=2.0, y=1.0, z=0.5, roll=0.2, pitch=-0.1, yaw=0.3),
    )

    x, y, z, qx, qy, qz, qw = pose
    assert x == pytest.approx(9.0)
    assert y == pytest.approx(22.0)
    assert z == pytest.approx(1.5)
    assert math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) == pytest.approx(1.0)
    expected_yaw = math.pi / 2 + 0.3
    actual_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    assert actual_yaw == pytest.approx(expected_yaw)


def test_shape_center_respects_configured_yaw_transform() -> None:
    x, y, *_ = compose_shape_center_pose(
        ObjectKinematicData(x=4.0, y=5.0, yaw=math.pi / 2),
        ShapeCenterPoseData(x=2.0),
        yaw_sign=-1.0,
    )

    assert x == pytest.approx(4.0)
    assert y == pytest.approx(3.0)
