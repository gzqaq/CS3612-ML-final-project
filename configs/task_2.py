from . import VAEConfig, BlockConfig

import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal, zeros_init


def get_config() -> VAEConfig:
  conv_pool_config = BlockConfig(
      features=None,
      conv_kernel_size=(3, 3),
      conv_strides=1,
      conv_padding="SAME",
      dtype=jnp.float32,
      kernel_init=orthogonal(np.sqrt(2)),
      pool_window_shape=(2, 2),
      pool_strides=(2, 2),
      pool_padding="VALID",
  )

  model_config = VAEConfig(
      latent_dim=512,
      image_size=32,
      dtype=jnp.float32,
      kernel_init=orthogonal(np.sqrt(2)),
      bias_init=zeros_init(),
      dropout_rate=0.1,
      conv=conv_pool_config,
  )

  return model_config
