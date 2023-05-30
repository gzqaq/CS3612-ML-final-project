from . import VAEConfig

import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal, zeros_init


def get_config() -> VAEConfig:
  model_config = VAEConfig(
    latent_dim=8192,
    image_size=250,
    dtype=jnp.float32,
    kernel_init=orthogonal(np.sqrt(2)),
    bias_init=zeros_init(),
    dropout_rate=0.1,
  )

  return model_config