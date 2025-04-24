# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from diffusers.models import AutoencoderKL

import torch
import torch.nn as nn

from flextok.utils.misc import str_to_dtype

VAE_BASE_CFG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.18.0.dev0",
    "_name_or_path": ".",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "in_channels": 3,
    "latent_channels": None,  # Overwritten by specific config
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": None,  # Overwritten by specific loaded model
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
}

__all__ = ["StableDiffusionVAE"]


class StableDiffusionVAE(nn.Module):
    """Wrapper for the AutoencoderKL class. This module wraps the VAE so that it matches the FlexTok
    API style for encoding, decoding, and autoencoding images.

    Args:
        images_read_key: Dictionary entry to read input images from.
        vae_latents_read_key: Dictionary entry to read VAE latents from.
        vae_latents_write_key: Dictionary entry which the VAE latents are written to.
        images_reconst_write_key: Dictionary entry which the reconstructions are written to.
        vae_kl_loss_write_key: Dictionary entry which the KL losses are written to. Defaults to None.
        latent_channels: Number of latent channels. Defaults to 16.
        scaling_factor: Latent scaling factor. Defaults to None.
        dtype_override: Dtype override value. Defaults to "fp32".
        hf_hub_path: HuggingFace hub path for AutoencoderKL-compatible model, e.g.
            'stabilityai/sdxl-vae'. Defaults to None.
        sample_posterior: Set to True to return x = mean + std * N(0,1), and False for x = mean.
            Defaults to True.
        learnable_logvar: Whether to use a learnable logvar. Defaults to False.
        logvar_init: logvar initialisation value. Defaults to 0.0.
        compile_encode_fn: Whether to compile the encoder modules. Defaults to False.
        force_vae_encode: If True, the VAE latent will always be computed and overwrite any existing
            pre-computed latents. If False, VAE encoder will not be run if the VAE latents already
            exist in the data_dict. Defaults to True.
        frozen: Wheather to freeze the model. Defaults to False.
    """

    def __init__(
        self,
        images_read_key: str,
        vae_latents_read_key: str,
        vae_latents_write_key: str,
        images_reconst_write_key: str,
        vae_kl_loss_write_key: Optional[str] = None,
        latent_channels: int = 16,
        scaling_factor: Optional[float] = None,
        dtype_override: Optional[str] = "fp32",
        hf_hub_path: Optional[str] = None,
        sample_posterior: bool = True,
        learnable_logvar: bool = False,
        logvar_init: float = 0.0,
        compile_encode_fn: bool = False,
        force_vae_encode: bool = True,
        frozen: bool = False,
    ):
        super().__init__()
        self.images_read_key = images_read_key
        self.vae_latents_read_key = vae_latents_read_key
        self.vae_latents_write_key = vae_latents_write_key
        self.images_reconst_write_key = images_reconst_write_key
        self.vae_kl_loss_write_key = vae_kl_loss_write_key

        if hf_hub_path is not None:
            # Optionally load existing VAE from HuggingFace
            self.vae = AutoencoderKL.from_pretrained(hf_hub_path, low_cpu_mem_usage=False)
        else:
            # Initialize a VAE from scratch
            vae_config = VAE_BASE_CFG.copy()
            vae_config["latent_channels"] = latent_channels
            self.vae = AutoencoderKL.from_config(vae_config)
        self.scaling_factor = scaling_factor or self.vae.config.scaling_factor

        if compile_encode_fn:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            self.vae.to(memory_format=torch.channels_last)
            self.vae.fuse_qkv_projections()  # Call this before freezing
            self.vae.encode = torch.compile(
                self.vae.encode, mode="max-autotune", fullgraph=True, dynamic=True
            )
        self.dtype_override = (
            torch.float32 if dtype_override is None else str_to_dtype(dtype_override)
        )
        self.sample_posterior = sample_posterior
        self.logvar: Optional[nn.Module] = None
        if learnable_logvar:
            # This is usually in the loss but we move it here so that it's part of the autoencoder's parameters.
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.force_vae_encode = force_vae_encode
        self.frozen = frozen
        if self.frozen:
            self.freeze()

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.vae.config["down_block_types"]) - 1)

    @property
    def latent_dim(self) -> int:
        return self.vae.config["latent_channels"]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def device_type(self) -> str:
        return self.device.type

    def init_weights_muP(self):
        # There are no muP modules to init.
        pass

    def freeze(self) -> "StableDiffusionVAE":
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def train(self, mode: bool = True) -> "StableDiffusionVAE":
        """Ignore, model is always frozen and in eval mode."""
        if self.frozen:
            # Ignore, model is always frozen and in eval mode.
            return self
        # If not frozen then use the normal train method
        return super().train(mode=mode)

    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encode images to the VAE latent space.

        After optional sampling from the posterior the latents are normalised by the scaling factor.

        Args:
            data_dict: Input data containing the following keys:
                - ``'{self.images_read_key}'`` - Images. Shape [B, 3, H, W].
                and optionally:
                - ``'{self.vae_latents_read_key}'`` - Pre-computed VAE latents. Shape [B, C, H, W].

        Returns:
            data_dict: Output data containing the following keys:
                - ``'{self.vae_latents_write_key}'`` The VAE latents. Shape [B, C, H, W].
                - ``'{self.vae_kl_loss_write_key}'`` The KL loss. Shape [B,]
        """
        if self.vae_latents_write_key in data_dict and not self.force_vae_encode:
            # If latents are already in data_dict, (optionally) skip encoding.
            return data_dict

        images = data_dict[self.images_read_key]

        is_list = False
        if isinstance(images, (list, tuple)):
            # Try to concatenate lists of tensors. This will throw an error if image sizes are different.
            images = torch.cat(images, dim=0)
            is_list = True

        with torch.amp.autocast(
            self.device_type,
            dtype=self.dtype_override,
            enabled=self.dtype_override != torch.float32,
        ):
            latent_dist = self.vae.encode(images.to(dtype=self.dtype_override)).latent_dist
            latents = latent_dist.sample() if self.sample_posterior else latent_dist.mode()
            latents *= self.scaling_factor
            kl_loss = latent_dist.kl()

        if is_list:
            latents = list(latents.split(1, dim=0))
            kl_loss = list(kl_loss.split(1, dim=0))

        data_dict[self.vae_latents_write_key] = latents
        data_dict[self.vae_kl_loss_write_key] = kl_loss
        return data_dict

    def decode(self, data_dict: Dict[str, Any], **ignore_kwargs) -> Dict[str, Any]:
        """Decode VAE latents to images.

        Args:
            data_dict: Input data containing the following keys:
                - ``'{self.vae_latents_read_key}'`` - VAE latents. Shape [B, C, H, W].

        Returns:
            data_dict: Output data containing the following keys:
                - ``'{self.images_reconst_write_key}'`` - Image reconstructions. Shape [B, 3, H, W].
        """
        latents = data_dict[self.vae_latents_read_key]

        is_list = False
        if isinstance(latents, (list, tuple)):
            # Try to concatenate lists of tensors. This will throw an error if image sizes are different.
            latents = torch.cat(latents, dim=0)
            is_list = True

        with torch.amp.autocast(
            self.device_type,
            dtype=self.dtype_override,
            enabled=self.dtype_override != torch.float32,
        ):
            images = self.vae.decode(
                latents.to(dtype=self.dtype_override) / self.scaling_factor
            ).sample

        if is_list:
            images = list(images.split(1, dim=0))

        data_dict[self.images_reconst_write_key] = images
        return data_dict

    def autoencode(self, data_dict: Dict[str, Any], **ignore_kwargs) -> Dict[str, Any]:
        """Autoencode images to reconstructions.

        Args:
            data_dict: Input data containing the following keys:
                - ``'{self.images_read_key}'`` - Images. Shape [B, 3, H, W].
                and optionally:
                - ``'{self.vae_latents_read_key}'`` - Pre-computed VAE latents. Shape [B, C, H, W].

        Returns:
            data_dict: Output data containing the following keys:
                - ``'{self.images_reconst_write_key}'`` - Image reconstructions. Shape [B, 3, H, W].
                - ``'{self.vae_latents_write_key}'`` The VAE latents. Shape [B, C, H, W].
                - ``'{self.vae_kl_loss_write_key}'`` The KL loss. Shape [B,]
        """
        data_dict = self.encode(data_dict)
        data_dict[self.vae_latents_read_key] = data_dict[self.vae_latents_write_key]
        return self.decode(data_dict)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self.autoencode(data_dict)

    def get_last_layer(self) -> torch.Tensor:
        return self.vae.decoder.conv_out.weight

    def get_logvar(self) -> Optional[torch.Tensor]:
        return self.logvar
