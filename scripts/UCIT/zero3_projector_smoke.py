#!/usr/bin/env python3
"""Small distributed regression test for ZeRO-3 projector weight loading."""

import argparse

import deepspeed
import torch

from llava.model.llava_arch import load_mm_projector_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.parse_args()

    deepspeed.init_distributed()
    config = {
        "train_batch_size": 2,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu"},
        },
    }
    with deepspeed.zero.Init(config_dict_or_path=config):
        projector = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 8),
        )
        state = {
            "0.weight": torch.full((8, 4), 0.25),
            "0.bias": torch.full((8,), 0.25),
            "2.weight": torch.full((8, 8), 0.25),
            "2.bias": torch.full((8,), 0.25),
        }
        load_mm_projector_state(projector, state)

    parameters = list(projector.parameters())
    with deepspeed.zero.GatheredParameters(parameters):
        for parameter in parameters:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))
    if torch.distributed.get_rank() == 0:
        print("ZeRO-3 projector load passed")


if __name__ == "__main__":
    main()
