from commvla.launch import build_command


def test_build_torchrun_command() -> None:
    command, environment = build_command(
        {
            "module": "commvla.training.train_dual_arm",
            "processes": 4,
            "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            "arguments": {
                "batch_size": 3,
                "freeze_vision_backbone": True,
                "no_freeze_llm_backbone": True,
                "unused": None,
            },
        }
    )
    assert "--nproc_per_node=4" in command
    assert command[-4:] == [
        "--batch-size",
        "3",
        "--freeze-vision-backbone",
        "--no-freeze-llm-backbone",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

