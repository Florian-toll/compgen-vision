# Copyright (c) 2025 Florian Redhardt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Models module for compositional generalization analysis.

This module contains model wrapper classes and utility functions for loading
and interacting with diffusion models.
"""

import logging
import os
from enum import Enum
from enum import auto
from typing import Any
from typing import Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from PIL import Image

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
  """Enum for supported model types."""

  STABLE_DIFFUSION = auto()
  STABLE_DIFFUSION_XL = auto()
  STABLE_DIFFUSION_3 = auto()
  FLUX = auto()
  UNKNOWN = auto()


def get_model_type(model_name: str) -> ModelType:
  """
  Determine model type from model name.

  Args:
      model_name: Name of the model

  Returns:
      ModelType enum value
  """
  model_name_lower = model_name.lower()

  if 'flux' in model_name_lower:
    return ModelType.FLUX
  elif 'stable-diffusion-xl' in model_name_lower:
    return ModelType.STABLE_DIFFUSION_XL
  elif any(x in model_name_lower for x in ['stable-diffusion-3', 'stable-diffusion-3.5']):
    return ModelType.STABLE_DIFFUSION_3
  elif 'stable-diffusion' in model_name_lower:
    return ModelType.STABLE_DIFFUSION
  else:
    return ModelType.UNKNOWN


CURRENT_STEP = None


def _step_callback(*args: Any, **kwargs: Any) -> dict:
  """
  Compatible with *both* Diffusers signatures:

  - ≤ 0.25 : (step:int, timestep:Tensor, latents:Tensor)
  - ≥ 0.26 : (pipe:DiffusionPipeline, step:int, timestep:Tensor,
             callback_kwargs:dict)

  Merely records the current **iteration index** in the global
  `CURRENT_STEP` and returns an (empty) dict so the pipeline is happy.
  """
  global CURRENT_STEP

  if not args:
    return {}

  if not isinstance(args[0], (int, float, torch.Tensor)):
    raw_step = args[1]
  else:
    raw_step = args[0]

  if torch.is_tensor(raw_step):
    raw_step = raw_step.item()

  CURRENT_STEP = int(raw_step)
  return {}


class ActivationHook:
  """
  Collect activations but pool first.

  • (B,C,H,W) ➜ global-avg-pool ➜ (B,C)
  • (B,N,D)   ➜ CLS / first token ➜ (B,D)

  Nothing else is changed - you still store **all** timesteps
  and use the same dtype as before.
  """

  def __init__(self, module, layer_name):
    self.layer_name = layer_name
    self._buf: dict[int, torch.Tensor] = {}
    self.hook = module.register_forward_hook(self._fn)

  def _fn(self, _module, _inp, out):
    global CURRENT_STEP
    if CURRENT_STEP is None:
      return

    if isinstance(out, tuple):
      out = out[0]
    if not torch.is_tensor(out):
      if hasattr(out, 'last_hidden_state'):
        out = out.last_hidden_state
      elif hasattr(out, 'hidden_states') and out.hidden_states:
        out = out.hidden_states[-1]
      else:
        return

    out = out.detach()
    out = DiffusionModelWrapper._pool_tensor(out)
    out = torch.from_numpy(out)

    self._buf[CURRENT_STEP] = out

  def tensor_at(self, t: int):
    return self._buf.get(t, None)

  def tensors_at(self, t_list: list[int]):
    return [self._buf[t] for t in t_list if t in self._buf]

  def remove(self):
    self.hook.remove()


class DiffusionModelWrapper:
  """Wrapper for diffusion models to capture and store activations."""

  def __init__(
    self,
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    token: Optional[str] = None,
  ):
    """
    Initialize the model wrapper.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache models
        device: Device to use for inference
        token: HuggingFace token for accessing gated models
    """
    self.model_name = model_name
    self.device = device
    self.hooks = {}
    self.model_type = get_model_type(model_name)
    self._default_fractions = [0.3, 0.5, 0.7, 1.0]

    logger.info(f'Loading model {model_name}...')
    logger.info(f'Cache directory: {cache_dir}')
    logger.info(f'Token available: {token is not None}')

    hf_cache = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
    logger.info(f'HF_HOME/TRANSFORMERS_CACHE environment variable: {hf_cache}')

    self._load_model(cache_dir, token)
    self.setup_hooks()

  def _load_model(self, cache_dir: Optional[str], token: Optional[str]) -> None:
    """
    Load the model based on its type.

    Args:
        cache_dir: Directory to cache models
        token: HuggingFace token for accessing gated models
    """
    dtype = torch.float16 if self.device == 'cuda' else torch.float32

    # Handle different model types
    if self.model_type == ModelType.FLUX:
      self._load_flux_model(cache_dir, dtype, token)
    elif self.model_type == ModelType.STABLE_DIFFUSION_XL:
      self._load_sdxl_model(cache_dir, dtype, token)
    elif self.model_type == ModelType.STABLE_DIFFUSION_3:
      self._load_sd3_model(cache_dir, dtype, token)
    elif self.model_type == ModelType.STABLE_DIFFUSION:
      self._load_sd_model(cache_dir, dtype, token)
    else:
      logger.warning(
        f'Unknown model type for {self.model_name}, trying standard pipeline'
      )

      self.pipeline = DiffusionPipeline.from_pretrained(
        self.model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        token=token,
      ).to(self.device)

    if hasattr(self.pipeline, 'enable_attention_slicing'):
      self.pipeline.enable_attention_slicing()

    self._print_model_architecture()

  def _print_model_architecture(self) -> None:
    """Print the complete model architecture recursively for any model type."""
    logger.info(f'\n=== Full {self.model_type} Architecture ===')

    def print_model_architecture(model, prefix='', depth=0):
      model_type = model.__class__.__name__
      num_params = sum(p.numel() for p in model.parameters())
      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

      if num_params > 0:
        logger.info(
          f'{prefix}├─ {model_type} [params: {num_params:,} ({trainable_params:,} trainable)]'
        )
      else:
        logger.info(f'{prefix}├─ {model_type}')

      children = list(model.named_children())
      for i, (name, child) in enumerate(children):
        if i == len(children) - 1:
          child_prefix = prefix + '│  '
          branch = '└─'
        else:
          child_prefix = prefix + '│  '
          branch = '├─'

        logger.info(f'{prefix}{branch} {name}')

        print_model_architecture(child, child_prefix, depth + 1)

    if self.model_type == ModelType.FLUX and hasattr(self.pipeline, 'transformer'):
      logger.info('\n=== Flux Transformer Structure ===')
      print_model_architecture(self.pipeline.transformer)

      for name, module in self.pipeline.transformer.named_children():
        logger.info(f'  {name}')

    elif self.model_type in [ModelType.STABLE_DIFFUSION, ModelType.STABLE_DIFFUSION_XL]:
      if hasattr(self.pipeline, 'unet'):
        logger.info('\n=== SD1/1.5 Structure ===')
        print_model_architecture(self.pipeline.unet)

    elif self.model_type == ModelType.STABLE_DIFFUSION_3:
      model = getattr(self.pipeline, 'transformer', None) or getattr(
        self.pipeline, 'unet', None
      )
      if model is not None:
        logger.info('\n=== SD3 Structure ===')
        print_model_architecture(model)

    if hasattr(self.pipeline, 'text_encoder'):
      logger.info('\n=== Text Encoder Structure ===')
      print_model_architecture(self.pipeline.text_encoder)

    if hasattr(self.pipeline, 'text_encoder_2'):
      logger.info('\n=== Text Encoder 2 Structure ===')
      print_model_architecture(self.pipeline.text_encoder_2)

  def _load_flux_model(
    self, cache_dir: Optional[str], dtype: torch.dtype, token: Optional[str]
  ) -> None:
    """Load a FLUX model."""
    logger.info(f'Loading FLUX model: {self.model_name}')

    pipeline_types = ['FluxPipeline', 'AutoPipelineForText2Image', 'DiffusionPipeline']

    for pipeline_type in pipeline_types:
      try:
        logger.info(f'Trying to load with {pipeline_type}...')
        module = __import__('diffusers', fromlist=[pipeline_type])
        pipeline_class = getattr(module, pipeline_type)

        self.pipeline = pipeline_class.from_pretrained(
          self.model_name,
          cache_dir=cache_dir,
          torch_dtype=dtype,
          use_safetensors=True,
          token=token,
        ).to(self.device)

        logger.info(f'Successfully loaded {self.model_name} with {pipeline_type}')
        return

      except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.debug(f'Failed to load with {pipeline_type}: {e}')
        continue

    raise ValueError(
      f'Failed to load FLUX model {self.model_name} with any pipeline type'
    )

  def _load_sdxl_model(
    self, cache_dir: Optional[str], dtype: torch.dtype, token: Optional[str]
  ) -> None:
    """Load a Stable Diffusion XL model."""
    logger.info(f'Loading SDXL model: {self.model_name}')

    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
      self.model_name,
      cache_dir=cache_dir,
      torch_dtype=dtype,
      token=token,
    ).to(self.device)

  def _load_sd3_model(
    self, cache_dir: Optional[str], dtype: torch.dtype, token: Optional[str]
  ) -> None:
    """Load a Stable Diffusion 3 model."""
    logger.info(f'Loading SD3 model: {self.model_name}')

    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
      self.model_name,
      cache_dir=cache_dir,
      torch_dtype=dtype,
      token=token,
    ).to(self.device)

  def _load_sd_model(
    self, cache_dir: Optional[str], dtype: torch.dtype, token: Optional[str]
  ) -> None:
    """Load a standard Stable Diffusion model."""
    logger.info(f'Loading SD model: {self.model_name}')

    self.pipeline = StableDiffusionPipeline.from_pretrained(
      self.model_name,
      cache_dir=cache_dir,
      torch_dtype=dtype,
      token=token,
    ).to(self.device)

  def setup_hooks(self) -> None:
    """Set up hooks to capture activations from key layers."""
    if self.model_type in [ModelType.STABLE_DIFFUSION, ModelType.STABLE_DIFFUSION_XL]:
      self._setup_unet_hooks()
    elif self.model_type == ModelType.FLUX:
      self._setup_flux_hooks()
    elif self.model_type == ModelType.STABLE_DIFFUSION_3:
      self._setup_sd3_hooks()

    self._setup_text_encoder_hooks()

    logger.info(f'Set up {len(self.hooks)} hooks for {self.model_type} model')

  def _setup_unet_hooks(self) -> None:
    """Set up hooks for all components of UNet-based models (SD and SDXL)."""
    if not hasattr(self.pipeline, 'unet'):
      logger.warning('No UNet found in model')
      return

    unet = self.pipeline.unet

    if hasattr(unet, 'conv_in'):
      self.hooks['conv_in'] = ActivationHook(unet.conv_in, 'conv_in')

    if hasattr(unet, 'time_embedding'):
      self.hooks['time_embedding'] = ActivationHook(unet.time_embedding, 'time_embedding')
    if hasattr(unet, 'add_embedding'):
      self.hooks['add_embedding'] = ActivationHook(unet.add_embedding, 'add_embedding')

    for i, block in enumerate(unet.down_blocks):
      if hasattr(block, 'resnets') and block.resnets is not None:
        for j, layer in enumerate(block.resnets):
          layer_name = f'down_{i}_res_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

      if hasattr(block, 'attentions') and block.attentions is not None:
        for j, layer in enumerate(block.attentions):
          layer_name = f'down_{i}_attn_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

          if (
            hasattr(layer, 'transformer_blocks') and layer.transformer_blocks is not None
          ):
            for k, transformer in enumerate(layer.transformer_blocks):
              # Self-attention
              if hasattr(transformer, 'attn1'):
                layer_name = f'down_{i}_self_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn1, layer_name)

              # Cross-attention
              if hasattr(transformer, 'attn2'):
                layer_name = f'down_{i}_cross_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn2, layer_name)

              # Feed-forward
              if hasattr(transformer, 'ff'):
                layer_name = f'down_{i}_ff_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.ff, layer_name)

      if hasattr(block, 'downsamplers') and block.downsamplers is not None:
        for j, layer in enumerate(block.downsamplers):
          layer_name = f'down_{i}_downsample_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

    if hasattr(unet, 'mid_block'):
      if hasattr(unet.mid_block, 'resnets') and unet.mid_block.resnets is not None:
        for j, layer in enumerate(unet.mid_block.resnets):
          layer_name = f'mid_res_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

      # Attentions
      if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        for j, layer in enumerate(unet.mid_block.attentions):
          layer_name = f'mid_attn_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

          if (
            hasattr(layer, 'transformer_blocks') and layer.transformer_blocks is not None
          ):
            for k, transformer in enumerate(layer.transformer_blocks):
              # Self-attention
              if hasattr(transformer, 'attn1'):
                layer_name = f'mid_self_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn1, layer_name)

              # Cross-attention
              if hasattr(transformer, 'attn2'):
                layer_name = f'mid_cross_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn2, layer_name)

              # Feed-forward
              if hasattr(transformer, 'ff'):
                layer_name = f'mid_ff_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.ff, layer_name)

    for i, block in enumerate(unet.up_blocks):
      if hasattr(block, 'resnets') and block.resnets is not None:
        for j, layer in enumerate(block.resnets):
          layer_name = f'up_{i}_res_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

      if hasattr(block, 'attentions') and block.attentions is not None:
        for j, layer in enumerate(block.attentions):
          layer_name = f'up_{i}_attn_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

          if (
            hasattr(layer, 'transformer_blocks') and layer.transformer_blocks is not None
          ):
            for k, transformer in enumerate(layer.transformer_blocks):
              # Self-attention
              if hasattr(transformer, 'attn1'):
                layer_name = f'up_{i}_self_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn1, layer_name)

              # Cross-attention
              if hasattr(transformer, 'attn2'):
                layer_name = f'up_{i}_cross_attn_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.attn2, layer_name)

              # Feed-forward
              if hasattr(transformer, 'ff'):
                layer_name = f'up_{i}_ff_{j}_{k}'
                self.hooks[layer_name] = ActivationHook(transformer.ff, layer_name)

      # Upsamplers
      if hasattr(block, 'upsamplers') and block.upsamplers is not None:
        for j, layer in enumerate(block.upsamplers):
          layer_name = f'up_{i}_upsample_{j}'
          self.hooks[layer_name] = ActivationHook(layer, layer_name)

    # Output layer
    if hasattr(unet, 'conv_out'):
      self.hooks['conv_out'] = ActivationHook(unet.conv_out, 'conv_out')

  def _setup_flux_hooks(self) -> None:
    """Set up hooks for FLUX models (transformer-based) with reduced sampling rate."""
    if not hasattr(self.pipeline, 'transformer'):
      logger.warning('No transformer found in FLUX model')
      return

    # Sampling rate - controls how many blocks to hook
    BLOCK_SAMPLING_RATE = 4

    transformer = self.pipeline.transformer

    self.hooks['flux_transformer'] = ActivationHook(transformer, 'flux_transformer')
    hook_count = 1

    for comp_name in [
      'context_embedder',
      'x_embedder',
      'norm_out',
      'proj_out',
      'time_text_embed',
      'pos_embed',
    ]:
      if hasattr(transformer, comp_name):
        self.hooks[f'flux_{comp_name}'] = ActivationHook(
          getattr(transformer, comp_name), f'flux_{comp_name}'
        )
        hook_count += 1

    if hasattr(transformer, 'transformer_blocks'):
      blocks = transformer.transformer_blocks
      logger.info(f'Found {len(blocks)} transformer_blocks')

      indices_to_hook = [0]
      if len(blocks) > 1:
        indices_to_hook.append(len(blocks) - 1)
      indices_to_hook.extend(
        [
          i
          for i in range(len(blocks))
          if i % BLOCK_SAMPLING_RATE == 0 and i not in indices_to_hook
        ]
      )
      indices_to_hook.sort()

      for i in indices_to_hook:
        block = blocks[i]
        layer_name = f'flux_transformer_block_{i}'
        self.hooks[layer_name] = ActivationHook(block, layer_name)
        hook_count += 1

        for comp in [
          'attn',
          'ff',
          'ff_context',
          'mlp',
          'norm',
          'norm1',
          'norm2',
          'self_attn',
          'cross_attn',
        ]:
          if hasattr(block, comp):
            self.hooks[f'{layer_name}_{comp}'] = ActivationHook(
              getattr(block, comp), f'{layer_name}_{comp}'
            )
            hook_count += 1

    if hasattr(transformer, 'single_transformer_blocks'):
      blocks = transformer.single_transformer_blocks
      logger.info(f'Found {len(blocks)} single transformer blocks')

      indices_to_hook = [0]
      if len(blocks) > 1:
        indices_to_hook.append(len(blocks) - 1)
      indices_to_hook.extend(
        [
          i
          for i in range(len(blocks))
          if i % BLOCK_SAMPLING_RATE == 0 and i not in indices_to_hook
        ]
      )
      indices_to_hook.sort()

      for i in indices_to_hook:
        block = blocks[i]
        layer_name = f'flux_single_block_{i}'
        self.hooks[layer_name] = ActivationHook(block, layer_name)
        hook_count += 1

        for comp in [
          'attn',
          'mlp',
          'norm',
          'norm1',
          'norm2',
          'proj_mlp',
          'act_mlp',
          'proj_out',
        ]:
          if hasattr(block, comp):
            self.hooks[f'{layer_name}_{comp}'] = ActivationHook(
              getattr(block, comp), f'{layer_name}_{comp}'
            )
            hook_count += 1

    if hasattr(self.pipeline, 'text_encoder'):
      self.hooks['text_encoder'] = ActivationHook(
        self.pipeline.text_encoder, 'text_encoder'
      )
      hook_count += 1

    if hasattr(self.pipeline, 'text_encoder_2'):
      self.hooks['text_encoder_2'] = ActivationHook(
        self.pipeline.text_encoder_2, 'text_encoder_2'
      )
      hook_count += 1

    logger.info(
      f'Set up {hook_count} hooks for FLUX model (sampling rate: {BLOCK_SAMPLING_RATE})'
    )

  def _setup_sd3_hooks(self) -> None:
    """Set up hooks for SD3.5 models based on the actual architecture."""
    model = getattr(self.pipeline, 'transformer', None) or getattr(
      self.pipeline, 'unet', None
    )
    if model is None:
      logger.warning('No suitable transformer module found for SD3 model')
      return

    transformer_blocks = getattr(model, 'transformer_blocks', None)
    if transformer_blocks is None:
      logger.warning('No transformer_blocks found in the model')
      return

    logger.info('\n=== [SD3.5] Setting up JointTransformerBlock hooks ===')
    num_hooks = 0

    for i, block in enumerate(transformer_blocks):
      if block.__class__.__name__ != 'JointTransformerBlock':
        continue

      if hasattr(block, 'attn') and block.attn is not None:
        lname = f'sd3_block_{i}_attn'
        self.hooks[lname] = ActivationHook(block.attn, lname)
        num_hooks += 1

      if hasattr(block, 'attn2') and block.attn2 is not None:
        lname = f'sd3_block_{i}_attn2'
        self.hooks[lname] = ActivationHook(block.attn2, lname)
        num_hooks += 1

      if hasattr(block, 'ff') and block.ff is not None:
        lname = f'sd3_block_{i}_ff'
        self.hooks[lname] = ActivationHook(block.ff, lname)
        num_hooks += 1

      if hasattr(block, 'ff_context') and block.ff_context is not None:
        lname = f'sd3_block_{i}_ff_context'
        self.hooks[lname] = ActivationHook(block.ff_context, lname)
        num_hooks += 1

    if hasattr(model, 'proj_out') and model.proj_out is not None:
      self.hooks['sd3_proj_out'] = ActivationHook(model.proj_out, 'sd3_proj_out')
      num_hooks += 1

    logger.info(f'[SD3.5] Set up {num_hooks} hooks.')

  def _setup_text_encoder_hooks(self) -> None:
    """Set up hooks for text encoders."""
    if hasattr(self.pipeline, 'text_encoder'):
      self.hooks['text_encoder'] = ActivationHook(
        self.pipeline.text_encoder, 'text_encoder'
      )
    if hasattr(self.pipeline, 'text_encoder_2'):
      self.hooks['text_encoder_2'] = ActivationHook(
        self.pipeline.text_encoder_2, 'text_encoder_2'
      )

  def generate_image(
    self,
    prompt: str,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
  ) -> Image.Image:
    """
    Generate an image and memorise how many denoising iterations were used.

    • If `num_inference_steps` is passed → that value wins
    • else, try to re-use a value you stored on `self` earlier
    • else, fall back to *whatever the pipeline already has* (scheduler length)
    """
    if num_inference_steps is None:
      num_inference_steps = getattr(self, 'num_inference_steps', None)

    if num_inference_steps is None:
      try:
        num_inference_steps = len(self.pipeline.scheduler.timesteps)
      except Exception:
        num_inference_steps = 30
    self.num_inference_steps = num_inference_steps

    generator = (
      torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
    )

    # Build the *common* kwargs once — easier to keep in sync
    common = dict(generator=generator, callback_on_step_end=_step_callback)

    with torch.no_grad():
      if self.model_type == ModelType.STABLE_DIFFUSION_XL:
        image = self.pipeline(
          prompt=prompt,
          num_inference_steps=num_inference_steps,
          guidance_scale=7.5,
          **common,
        ).images[0]

      elif self.model_type == ModelType.FLUX:
        image = self.pipeline(
          prompt=prompt,
          num_inference_steps=num_inference_steps,
          guidance_scale=3.5,
          max_sequence_length=512,
          **common,
        ).images[0]

      elif self.model_type == ModelType.STABLE_DIFFUSION_3:
        image = self.pipeline(
          prompt=prompt,
          num_inference_steps=num_inference_steps,
          guidance_scale=7.5,
          **common,
        ).images[0]

      else:
        image = self.pipeline(
          prompt=prompt,
          num_inference_steps=num_inference_steps,
          **common,
        ).images[0]

    return image

  def _fractions_to_steps(self, fractions: list[float]) -> list[int]:
    """
    Convert user fractions (0-1) to step indices 0 … S-1.

    • S is taken from `self.num_inference_steps` *if available*
    • otherwise we look at the current scheduler length
    • final fallback: 30
    """
    S = getattr(self, 'num_inference_steps', None)
    if S is None:
      try:
        S = len(self.pipeline.scheduler.timesteps)
      except Exception:
        S = 30

    idxs = [min(int(f * (S - 1)), S - 2) for f in fractions]
    return sorted(dict.fromkeys(idxs))

  def get_activations(
    self,
    fractions: list[float] | None = None,
    steps: list[int] | None = None,
    agg: str = 'concat',
  ) -> dict[str, np.ndarray]:
    """
    Return {layer_name: feature-matrix} aggregated as requested.

    agg options
    -----------
    "concat"      - concatenate picked steps, shape (B, K·D)
    "mean"        - average over picked,      shape (B, D)
    "cat+mean"    - concat picked + their mean
    "cat+allmean" - concat picked + mean(all recorded)
    "none"        - stack   (B, K, D)
    """
    if steps is None:
      if fractions is None:
        fractions = self._default_fractions
      steps = self._fractions_to_steps(fractions)

    feats: dict[str, np.ndarray] = {}

    for lname, hook in self.hooks.items():
      if not hook._buf:
        continue

      all_steps = sorted(hook._buf.keys())
      all_feat = [self._pool_tensor(hook._buf[s]) for s in all_steps]

      picked_feat = [self._pool_tensor(hook._buf[s]) for s in steps if s in hook._buf]
      if not picked_feat:
        continue

      if agg == 'concat':
        feat = np.concatenate(picked_feat, 1)
      elif agg == 'mean':
        feat = np.mean(picked_feat, 0)
      elif agg == 'cat+mean':
        feat = np.concatenate(
          [np.concatenate(picked_feat, 1), np.mean(picked_feat, 0)], 1
        )
      elif agg == 'cat+allmean':
        feat = np.concatenate([np.concatenate(picked_feat, 1), np.mean(all_feat, 0)], 1)
      elif agg == 'none':
        feat = np.stack(picked_feat, 1)
      else:
        raise ValueError(f"unknown agg='{agg}'")

      feats[lname] = feat

      D = picked_feat[0].shape[-1]
      print(
        f'[DEBUG] {lname}: steps={steps} K={len(picked_feat)} '
        f'D={D} -> feat.shape={feat.shape}'
      )

    return feats

  @staticmethod
  def _pool_tensor(t: torch.Tensor) -> np.ndarray:
    """
    UNet conv / attention  :  (B, C, H, W)  → global-avg-pool → (B, C)
    Transformer sequence   :  (B, N, D)     → mean tokens  → (B, D)
    Already pooled         :  (B, D)        → unchanged
    """
    if t.ndim == 4:  # UNet
      t = t.mean(dim=(2, 3))  # H-W average
    elif t.ndim == 3:  # seq-length
      t = t.mean(dim=1)  # pool accros sequence
    return t.cpu().numpy()
