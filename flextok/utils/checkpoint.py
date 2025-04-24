# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
import ast
import io
import json
import os
from pathlib import Path
from typing import Optional, Union

from safetensors.torch import load as load_st
from safetensors.torch import save_file

import torch

from hydra.utils import instantiate

__all__ = ["save_safetensors", "load_safetensors", "load_model_from_safetensors"]


def save_safetensors(state_dict, ckpt_path, metadata_dict=None):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
    if metadata_dict is not None:
        metadata = {k: str(v) for k, v in metadata_dict.items()}
    else:
        metadata = None
    save_file(state_dict, ckpt_path, metadata=metadata)


def parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        try:
            v_parsed = ast.literal_eval(v)
        except:
            v_parsed = v
        metadata[k] = v_parsed
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, "rb") as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors

    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})
    metadata = parse_metadata(metadata)

    return tensors, metadata


def load_model_from_safetensors(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    to_eval: bool = True,
) -> torch.nn.Module:
    """Loads a safetensors checkpoint from the given path and instantiates
    a model from the config safed in it.

    Args:
        ckpt_path: Path to .safetensors checkpoint
        device: Optional torch device
        to_eval: Set to call .eval() on model

    Returns:
        Model with loaded weights.
    """
    ckpt, config = load_safetensors(ckpt_path)
    model = instantiate(config)
    model.load_state_dict(ckpt)

    if device is not None:
        model = model.to(device)

    if to_eval:
        model = model.eval()

    return model
