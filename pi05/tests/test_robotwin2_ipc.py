import socket

import numpy as np
import pytest

from pi05_fabric.evaluation.robotwin2_ipc import decode_npz_message
from pi05_fabric.evaluation.robotwin2_ipc import deterministic_smoke_response
from pi05_fabric.evaluation.robotwin2_ipc import encode_npz_message
from pi05_fabric.evaluation.robotwin2_ipc import paired_flow_seed
from pi05_fabric.evaluation.robotwin2_ipc import receive_message
from pi05_fabric.evaluation.robotwin2_ipc import send_message
from pi05_fabric.evaluation.robotwin2_ipc import validate_policy_response


def _request():
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "kind": np.asarray("inference"),
        "request_id": np.asarray("17:2"),
        "condition_id": np.asarray(17, dtype=np.int64),
        "planning_call": np.asarray(2, dtype=np.int64),
        "instruction": np.asarray("handover the red block"),
        "global_image": np.zeros((12, 16, 3), dtype=np.uint8),
        "left_wrist_image": np.ones((12, 16, 3), dtype=np.uint8),
        "right_wrist_image": np.full((12, 16, 3), 2, dtype=np.uint8),
        "eef_state": np.zeros((20,), dtype=np.float32),
    }


def test_npz_codec_does_not_require_pickle():
    framed = encode_npz_message(_request())
    decoded = decode_npz_message(framed[8:])
    assert decoded["request_id"].item() == "17:2"
    assert decoded["global_image"].dtype == np.uint8


def test_socket_round_trip_and_deterministic_response():
    left, right = socket.socketpair()
    try:
        send_message(left, _request())
        received = receive_message(right)
        response_a = deterministic_smoke_response(received)
        response_b = deterministic_smoke_response(received)
        np.testing.assert_array_equal(response_a["actions_r6"], response_b["actions_r6"])
        send_message(right, response_a)
        actions = validate_policy_response(receive_message(left), request_id="17:2")
        assert actions.shape == (50, 20)
    finally:
        left.close()
        right.close()


def test_response_rejects_mismatched_request_id():
    response = deterministic_smoke_response(_request())
    with pytest.raises(ValueError, match="request_id"):
        validate_policy_response(response, request_id="other")


def test_paired_flow_seed_is_stable_and_planning_call_specific():
    assert paired_flow_seed(17, 2) == paired_flow_seed(17, 2)
    assert paired_flow_seed(17, 2) != paired_flow_seed(17, 3)
    with pytest.raises(ValueError, match="non-negative"):
        paired_flow_seed(-1, 0)
