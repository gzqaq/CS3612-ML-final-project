from basic_types import Tuple, Callable, KeyArray
from configs import VAEConfig, ModelConfig, BlockConfig

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


def conv(config: BlockConfig) -> nn.Module:
  return nn.Conv(features=config.features,
                 kernel_size=config.conv_kernel_size,
                 strides=config.conv_strides,
                 padding=config.conv_padding,
                 dtype=config.dtype,
                 kernel_init=config.kernel_init,
                 use_bias=False)

def batchnorm(config: BlockConfig, train: bool = True) -> nn.Module:
  return nn.BatchNorm(use_running_average=not train,
                      momentum=0.9,
                      epsilon=1e-5,
                      dtype=config.dtype)

def maxpool(config: BlockConfig) -> Callable:
  return partial(nn.max_pool,
                 window_shape=config.pool_window_shape,
                 strides=config.pool_strides,
                 padding=config.pool_padding)

def reparameterize(rng: KeyArray, mean: jax.Array, logvar: jax.Array) -> jax.Array:
  std = jnp.exp(0.5 * logvar)
  eps = jax.random.normal(rng, logvar.shape)

  return mean + eps * std


class ConvPoolBlock(nn.Module):
  config: BlockConfig
  n_convs: int
  n_features: int

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    config = self.config.replace(features=self.n_features)
    for _ in range(self.n_convs):
      inp = conv(config)(inp)
      inp = batchnorm(config, train)(inp)
      inp = nn.relu(inp)

    return maxpool(config)(inp)
  

class ConvResBlock(nn.Module):
  config: BlockConfig
  n_convs: int
  n_features: int

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    assert inp.shape[-1] == self.n_features
    config = self.config.replace(features=self.n_features)

    for _ in range(self.n_convs):
      x = conv(config)(inp)
      x = batchnorm(config, train)(x)
      x = nn.relu(x)

    return inp + x
  

class Encoder(nn.Module):
  config: VAEConfig

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> Tuple[jax.Array, jax.Array]:
    dense = partial(nn.Dense,
                    dtype=self.config.dtype,
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init)
    dropout = nn.Dropout(rate=self.config.dropout_rate,
                         deterministic=not train)
    
    inp = inp.reshape(inp.shape[0], -1)
    inp = nn.relu(dropout(dense(4096)(inp)))
    inp = nn.relu(dropout(dense(2048)(inp)))

    mean = dense(self.config.latent_dim)(inp)
    logvar = dense(self.config.latent_dim)(inp)

    return mean, logvar


class Decoder(nn.Module):
  config: VAEConfig

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    dense = partial(nn.Dense,
                    dtype=self.config.dtype,
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init)
    dropout = nn.Dropout(rate=self.config.dropout_rate,
                         deterministic=not train)
    
    inp = nn.relu(dropout(dense(2048)(inp)))
    inp = nn.relu(dropout(dense(4096)(inp)))
    inp = dense(self.config.image_size ** 2 * 3)(inp)

    return inp.reshape(-1, self.config.image_size, self.config.image_size, 3)


class VAE(nn.Module):
  config: VAEConfig

  def setup(self):
    self.encoder = Encoder(self.config)
    self.decoder = Encoder(self.config)

  def encode(self, inp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    return self.encoder(inp, False)
  
  def decode(self, inp: jax.Array) -> jax.Array:
    return self.decoder(inp, False)
  
  def __call__(self, reparam_rng: KeyArray, inp: jax.Array, train: bool = True) -> jax.Array:
    mean, logvar = self.encoder(inp, train)
    z = reparameterize(reparam_rng, mean, logvar)
    output = self.decoder(z)

    return output, mean, logvar


class MyModel(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    conv_pool_config = self.config.conv_pool
    conv_res_config = self.config.conv_res

    dense = partial(nn.Dense,
                    dtype=self.config.dtype,
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init)
    dropout = nn.Dropout(rate=self.config.dropout_rate,
                         deterministic=not train)

    inp = ConvPoolBlock(conv_pool_config, 2, 8)(inp, train)
    inp = ConvPoolBlock(conv_pool_config, 2, 64)(inp, train)
    inp = ConvPoolBlock(conv_pool_config, 3, 128)(inp, train)
    inp = ConvResBlock(conv_res_config, 3, 128)(inp, train)
    inp = ConvPoolBlock(conv_pool_config, 4, 256)(inp, train)
    inp = ConvResBlock(conv_res_config, 4, 256)(inp, train)

    inp = inp.reshape(inp.shape[0], -1)

    inp = nn.relu(dropout(dense(4096)(inp)))
    inp = nn.relu(dropout(dense(2048)(inp)))
    inp = nn.relu(dropout(dense(1024)(inp)))

    return dense(self.config.out_features)(inp)