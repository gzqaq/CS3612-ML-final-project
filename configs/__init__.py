import jax.numpy as jnp
from flax import struct
from typing import Any, Optional, Tuple, Union


@struct.dataclass
class BlockConfig:
  features: Optional[int]
  conv_kernel_size: Tuple[int, int]
  conv_strides: int
  conv_padding: str
  dtype: Any
  kernel_init: Any
  pool_window_shape: Tuple[int]
  pool_strides: Tuple[int]
  pool_padding: str


@struct.dataclass
class ModelConfig:
  out_features: int
  dtype: Any
  kernel_init: Any
  bias_init: Any
  dropout_rate: float
  conv_pool: BlockConfig
  conv_res: BlockConfig

@struct.dataclass
class VAEConfig:
  latent_dim: int
  image_size: int
  dtype: Any
  kernel_init: Any
  bias_init: Any
  dropout_rate: float
  conv: BlockConfig

@struct.dataclass
class TrainConfig:
  run_name: str
  seed: int
  dtype: Any
  n_classes: int
  n_epochs: int
  batch_size: int
  lr: float
  clip_norm: float
  model: Union[ModelConfig, VAEConfig]


def parse_user_flags(flags) -> TrainConfig:
  if isinstance(flags.dtype, str):
    dtype = {"float32": jnp.float32,
                   "float64": jnp.float64,
                   "float16": jnp.float16,
                   "bfloat16": jnp.bfloat16}[flags.dtype]
  model_config = __import__(f"configs.{flags.model_config}", fromlist=[""]).get_config()

  config = TrainConfig(
    run_name=flags.run_name,
    seed=flags.seed,
    dtype=dtype,
    n_classes=flags.n_classes,
    n_epochs=flags.n_epochs,
    batch_size=flags.batch_size,
    lr=flags.lr,
    clip_norm=flags.clip_norm,
    model=model_config
  )
  print(config)

  return config