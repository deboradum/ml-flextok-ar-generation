# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from huggingface_hub import PyTorchModelHubMixin

import torch
import torch.nn as nn

from hydra.utils import instantiate

from .model.utils.packed_ops import packed_call


class FlexTok(nn.Module):
    """
    Main FlexTok tokenizer module. This module wraps the VAE, Encoder, Regularizer, Decoder, and Flow Matching
    modules and provides a high-level API for tokenizing, detokenizing, and autoencoding images.

    Args:
        vae: The VAE module used to perceptually compress images into a smaller latent space.
        encoder: The encoder module that maps the VAE latents into tokens.
        decoder: The decoder module that maps tokens back into images.
        regularizer: The regularizer module, e.g. an FSQ module, that quantizes the tokens.
        flow_matching_noise_module: The flow matching noise module that adds noise to the latents during training.
        pipeline: The flow matching pipeline module that performs the decoder denoising during inference.
    """

    def __init__(
        self,
        vae: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        regularizer: nn.Module,
        flow_matching_noise_module: nn.Module,
        pipeline: nn.Module,
    ):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.regularizer = regularizer

        self.token_write_key = self.regularizer.tokens_write_key
        self.quants_write_key = self.regularizer.quants_write_key
        self.image_write_key = self.vae.images_reconst_write_key

        self.flow_matching_noise_module = flow_matching_noise_module
        self.pipeline = pipeline(model=self.decoder)

    def init_weights_muP(self):
        self.encoder.init_weights_muP()
        self.decoder.init_weights_muP()

    @property
    def downsample_factor(self) -> int:
        return self.vae.downsample_factor

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict = self.encode(data_dict)  # VAE encode | Encoder | Regularizer
        # Adds noised image, timesteps, sigmas, weights.
        data_dict = self.flow_matching_noise_module(data_dict)
        data_dict = self.decoder(data_dict)  # Decoder
        return data_dict

    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encodes a dictionary of input images into tokens.

        Args:
            data_dict: A dictionary containing an entry 'rgb' with the
                input images being a List of tensors of shape [1, C, H, W].

        Returns:
            Dictionary containing the encoded tokens and other intermediate results.
        """
        with torch.no_grad():
            data_dict = self.vae.encode(data_dict)
        data_dict = self.encoder(data_dict)
        data_dict = self.regularizer(data_dict)
        return data_dict

    def decode(
        self,
        data_dict: Dict[str, Any],
        timesteps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        vae_image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        perform_norm_guidance: bool = False,
        **ignore_kwargs,
    ) -> Dict[str, Any]:
        """Decodes a dictionary of quantized tokens into images.

        Args:
            data_dict: A dictionary containing an entry self.quants_write_key
                with a list of the quantized tokens, each of shape [1, L, 6].
            timesteps: Number of inference denoising steps.
            generator: Optional torch.Generator to set seed for noise sampling.
            vae_image_sizes: VAE image sizes, needs to be given in terms of latent space dimensions.
                For example, 32 when using an f8 VAE on images of size 256x256.
            verbose: Whether to show tqdm progress bar.
            guidance_scale: Classifier-free guidance scale.
            perform_norm_guidance: Whether to perform APG, see https://arxiv.org/abs/2410.02416.

        Returns:
            Dictionary containing the decoded reconstructions and other intermediate results.
        """
        data_dict = self.pipeline(
            data_dict,
            generator=generator,
            timesteps=timesteps,
            vae_image_sizes=vae_image_sizes,
            verbose=verbose,
            guidance_scale=guidance_scale,
            perform_norm_guidance=perform_norm_guidance,
        )
        with torch.no_grad():
            data_dict = self.vae.decode(data_dict)
        return data_dict

    def autoencode(
        self,
        data_dict: Dict[str, Any],
        timesteps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        vae_image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        perform_norm_guidance: bool = False,
        **ignore_kwargs,
    ) -> Dict[str, Any]:
        """
        Encodes and decodes a dictionary of input images.
        See encode and decode functions for arguments.
        """
        data_dict = self.encode(data_dict)  # VAE encode | Encoder | Regularizer
        data_dict = self.decode(
            data_dict,
            timesteps=timesteps,
            generator=generator,
            vae_image_sizes=vae_image_sizes,
            verbose=verbose,
            guidance_scale=guidance_scale,
            perform_norm_guidance=perform_norm_guidance,
        )
        return data_dict

    def tokenize(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Tokenizes the input images into the full 1D sequences of discretized
        register tokens.

        Args:
            images: The input image tensor of shape [B, C, H, W].

        Returns:
            List of image token ids. Each of shape like [1, L].
        """
        # The encoder expects a list of [1, C, H, W] images.
        data_dict = {self.vae.images_read_key: images.split(1)}
        data_dict = self.encode(data_dict)
        token_ids_list = data_dict[self.token_write_key]
        return token_ids_list

    def _get_padded_token_seq(self, token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Pad the token sequence to the maximum length.

        Args:
            token_ids: The token id sequence of shape [1, l].
            max_seq_len: The maximum sequence length L.

        Returns:
            Padded token id sequence.
        """
        device, dtype = token_ids.device, token_ids.dtype
        pad_len = max_seq_len - token_ids.shape[1]
        pad_seq = torch.zeros((1, pad_len), device=device, dtype=dtype)
        return torch.cat([token_ids, pad_seq], dim=1)  # [1, L]

    def _prepare_data_dict_for_detokenization(
        self,
        token_ids_list: list[torch.Tensor],
    ) -> Dict[str, Any]:
        """Prepare a data_dict for decoding token ID lists to images.

        Args:
            token_ids_list: List of token id sequences.

        Returns:
            Data dict for decoding into images.
        """
        device = token_ids_list[0].device

        # Token id lists may be truncated. First, pad them to the maximum length.
        max_register_tokens = self.encoder.module_dict["enc_register_module"].n_max
        token_ids_lens = [t.shape[1] for t in token_ids_list]
        token_ids_list = [
            self._get_padded_token_seq(t, max_register_tokens) for t in token_ids_list
        ]

        # Look up the quantized embeddings for the discrete tokens.
        fsq_decode_fn = partial(self.regularizer.indices_to_embedding)
        quant_list = packed_call(fsq_decode_fn, token_ids_list)

        # Prepare the data dict for the decoder.
        eval_keep_k_read_key = self.decoder.module_dict["dec_nested_dropout"].eval_keep_k_read_key
        data_dict = {self.quants_write_key: quant_list, eval_keep_k_read_key: token_ids_lens}

        return data_dict

    def detokenize(
        self,
        token_ids_list: list[torch.Tensor],
        vae_image_sizes: int = 32,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Decode images from token ids.

        Args:
            token_ids_list: List of image tokens ids of shape [1, l]. Can be variable length.
            vae_image_sizes: The VAE grid size to decode. By default, 32, when using an f8 VAE
                on images of size 256x256.
            timesteps: Number of inference denoising steps.
            generator: Optional torch.Generator to set seed for noise sampling.
            verbose: Whether to show tqdm progress bar.
            guidance_scale: Classifier-free guidance scale.
            perform_norm_guidance: Whether to perform APG, see https://arxiv.org/abs/2410.02416.

        Returns:
            Tensor of decoded RGB images of shape [B, C, H, W].
        """
        # Map the data dict fields to those expected by the resampler decoder.
        data_dict = self._prepare_data_dict_for_detokenization(token_ids_list=token_ids_list)

        data_dict = self.decode(
            data_dict=data_dict,
            vae_image_sizes=vae_image_sizes,
            **kwargs,
        )

        decoded_images_list = data_dict[self.image_write_key]
        return torch.cat(decoded_images_list, dim=0)


class FlexTokFromHub(FlexTok, PyTorchModelHubMixin):
    """Wrapper around FlexTok for easy loading with Huggingface Hub.

    Args:
        config (dict): Dictionary containing the model configuration,
            used for loading from Huggingface Hub.
    """

    def __init__(self, config: dict):

        config = copy.deepcopy(config)

        super().__init__(
            vae=instantiate(config["vae"]),
            encoder=instantiate(config["encoder"]),
            decoder=instantiate(config["decoder"]),
            regularizer=instantiate(config["regularizer"]),
            flow_matching_noise_module=instantiate(config["flow_matching_noise_module"]),
            pipeline=instantiate(config["pipeline"]),
        )
