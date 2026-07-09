import math

import pytest
from pisa_api.av import (
    ObjectKinematicData,
    ObjectStateData,
    ObservationData,
    ObservedAgentData,
    ShapeCenterPoseData,
    ShapeData,
    ShapeDimensionData,
)

from autoware_wrapper.geometry import (
    ObservationContractError,
    ObservationNormalizer,
    clamp_ackermann_speed,
    compose_shape_center_pose,
    validate_ackermann_payload,
)


def _shape(*, length: float = 4.0) -> ShapeData:
    return ShapeData(dimensions=ShapeDimensionData(x=length, y=2.0, z=1.5))


def _state(x: float, *, time_ns: int = 0, shape: ShapeData | None = None) -> ObjectStateData:
    return ObjectStateData(
        kinematic=ObjectKinematicData(time_ns=time_ns, x=x),
        shape=shape,
    )


def test_observation_uses_explicit_ego_and_agent_states() -> None:
    ego = _state(100.0)
    first = _state(1.0, shape=_shape())
    second = _state(2.0, shape=_shape())
    observation = ObservationData(
        ego=ego,
        agents=[
            ObservedAgentData(state=first),
            ObservedAgentData(state=second, tracking_id=42, entity_name="privileged-name"),
        ],
    )

    actual_ego, agents = ObservationNormalizer().normalize(observation, 0)

    assert actual_ego is ego
    assert agents == [first, second]


def test_agent_permutation_does_not_change_ego_or_detected_state_set() -> None:
    ego = _state(100.0)
    first = ObservedAgentData(state=_state(1.0, shape=_shape()), tracking_id=11)
    second = ObservedAgentData(state=_state(2.0, shape=_shape()))
    normalizer = ObservationNormalizer()

    ego_a, agents_a = normalizer.normalize(ObservationData(ego=ego, agents=[first, second]), 0)
    ego_b, agents_b = normalizer.normalize(ObservationData(ego=ego, agents=[second, first]), 0)

    assert ego_a == ego_b == ego
    assert {agent.kinematic.x for agent in agents_a} == {agent.kinematic.x for agent in agents_b}


def test_shape_cache_survives_reorder_and_is_cleared_on_reset() -> None:
    normalizer = ObservationNormalizer()
    ego = _state(100.0)
    shape = _shape()
    normalizer.normalize(
        ObservationData(
            ego=ego,
            agents=[ObservedAgentData(state=_state(1.0, shape=shape), tracking_id=7)],
        ),
        0,
    )

    _, agents = normalizer.normalize(
        ObservationData(
            ego=_state(100.0, time_ns=1),
            agents=[ObservedAgentData(state=_state(1.0, time_ns=1), tracking_id=7)],
        ),
        1,
    )
    assert agents[0].shape == shape

    normalizer.reset()
    with pytest.raises(ObservationContractError, match="first observation"):
        normalizer.normalize(
            ObservationData(
                ego=_state(100.0, time_ns=2),
                agents=[ObservedAgentData(state=_state(1.0, time_ns=2), tracking_id=7)],
            ),
            2,
        )


def test_missing_shape_without_tracking_id_and_shape_mutation_are_rejected() -> None:
    normalizer = ObservationNormalizer()
    ego = _state(100.0)
    with pytest.raises(ObservationContractError, match="no tracking_id"):
        normalizer.normalize(
            ObservationData(ego=ego, agents=[ObservedAgentData(state=_state(1.0))]),
            0,
        )

    normalizer.normalize(
        ObservationData(
            ego=ego,
            agents=[ObservedAgentData(state=_state(1.0, shape=_shape()), tracking_id=8)],
        ),
        0,
    )
    with pytest.raises(ObservationContractError, match="changed static shape"):
        normalizer.normalize(
            ObservationData(
                ego=ego,
                agents=[
                    ObservedAgentData(state=_state(1.0, shape=_shape(length=5.0)), tracking_id=8)
                ],
            ),
            0,
        )


def test_shape_cache_accepts_numerically_equivalent_geometry() -> None:
    normalizer = ObservationNormalizer()
    ego = _state(100.0)
    normalizer.normalize(
        ObservationData(
            ego=ego,
            agents=[ObservedAgentData(state=_state(1.0, shape=_shape()), tracking_id=9)],
        ),
        0,
    )
    _, agents = normalizer.normalize(
        ObservationData(
            ego=ego,
            agents=[
                ObservedAgentData(
                    state=_state(1.0, shape=_shape(length=4.0 + 1e-10)),
                    tracking_id=9,
                )
            ],
        ),
        0,
    )
    assert agents[0].shape is not None


def test_shape_center_offset_and_orientation_are_composed_in_canonical_frame() -> None:
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


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_observation_values_are_rejected(value: float) -> None:
    with pytest.raises(ObservationContractError, match="non-finite"):
        ObservationNormalizer().normalize(
            ObservationData(ego=_state(value)),
            0,
        )


def test_timestamp_and_non_positive_dimensions_are_rejected() -> None:
    with pytest.raises(ObservationContractError, match="time_ns"):
        ObservationNormalizer().normalize(ObservationData(ego=_state(0.0, time_ns=3)), 4)
    with pytest.raises(ObservationContractError, match="dimensions"):
        ObservationNormalizer().normalize(
            ObservationData(
                ego=_state(0.0),
                agents=[ObservedAgentData(state=_state(1.0, shape=_shape(length=0.0)))],
            ),
            0,
        )


def test_ackermann_payload_validation() -> None:
    validate_ackermann_payload({"steer": 0.1, "speed": 3.0, "steer_speed": 0.2})

    for payload in (
        {"steer": 0.1, "speed": -1.0},
        {"steer": 0.1, "speed": 1.0, "steer_speed": -0.1},
        {"steer": math.nan, "speed": 1.0},
        {"steer": 0.1, "speed": 1.0, "unknown": 2.0},
    ):
        with pytest.raises(ValueError):
            validate_ackermann_payload(payload)


def test_negative_autoware_speed_is_clamped_to_zero() -> None:
    assert clamp_ackermann_speed(-2.5) == 0.0
    assert clamp_ackermann_speed(3.0) == 3.0
    assert math.isnan(clamp_ackermann_speed(math.nan))
