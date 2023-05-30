from basic_types import Array, Minibatch, Metric, KeyArray, Dict
from configs import TrainConfig, task_2
from model import MyModel, VAE
import vae_utils

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
from absl import logging
from flax.training import train_state, checkpoints
from typing import Callable, Tuple, Any


class TrainState(train_state.TrainState):
  batch_stats: Any


@jax.vmap
def kl_div(mean: jax.Array, logvar: jax.Array) -> jax.Array:
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
  logits = jax.nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def vae_train_step(runner_state: Tuple[TrainState, KeyArray], minibatch: jax.Array) -> Tuple[TrainState, KeyArray]:
  state, rng = runner_state
  rng, reparam_rng = jax.random.split(rng)

  def loss_fn(params):
    recon, mean, logvar = state.apply_fn(
      {"params": params},
      reparam_rng, minibatch, True
    )
    bce_loss = binary_cross_entropy_with_logits(recon, minibatch).mean()
    kld_loss = kl_div(mean, logvar).mean()
    
    return bce_loss + kld_loss, (bce_loss, kld_loss)
  
  (loss, aux_vals), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  train_metrics = {"loss": loss, "bce_loss": aux_vals[0], "kld_loss": aux_vals[1]}

  return (state.apply_gradients(grads=grads), rng), train_metrics

def vae_metrics(recon: jax.Array, image: jax.Array, mean: jax.Array, logvar: jax.Array) -> Dict[str, jax.Array]:
  bce_loss = binary_cross_entropy_with_logits(recon, image).mean()
  kld_loss = kl_div(mean, logvar).mean()

  return {"loss": bce_loss + kld_loss,
          "bce_loss": bce_loss,
          "kld_loss": kld_loss}

@jax.jit
def vae_eval(rng: KeyArray, state: TrainState, val_inp: jax.Array, latent_vec: jax.Array) -> jax.Array:
  def eval_model(vae_model: VAE):
    recon, mean, logvar = vae_model(rng, val_inp, False)
    comparison = jnp.concatenate([val_inp[:8].reshape(-1, 250, 250, 3),
                                  recon[:8].reshape(-1, 250, 250, 3)])
    gen_images = vae_model.decode(latent_vec)
    metrics = vae_metrics(recon, val_inp, mean, logvar)

    return metrics, comparison, gen_images
  
  return nn.apply(eval_model, VAE(task_2.get_config()))({"params": state.params})


def make_train(config: TrainConfig,
               train_set: Tuple[Array, Array],
               val_set: Tuple[Array, Array]) -> Callable:
  train_inputs, train_labels = train_set
  val_inputs, val_labels = val_set

  train_set_size = train_inputs.shape[0]
  minibatch_size = config.batch_size
  n_minibatches = train_set_size // minibatch_size
  
  train_inputs, val_inputs = jax.tree_map(lambda x: x.astype(config.dtype),
                                          (train_inputs, val_inputs))
  train_labels, val_labels = jax.tree_map(lambda x: x.astype(jnp.uint8),
                                          (train_labels, val_labels))

  def train(rng):
    model = MyModel(config.model)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           train_inputs[:1])
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    tx = optax.chain(optax.clip_by_global_norm(config.clip_norm),
                     optax.adamw(config.lr))
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats)
    del model, variables, params, batch_stats

    def _train_one_epoch(runner_state, _):
      train_state, rng = runner_state

      def _update_minibatch(runner_state: Tuple[TrainState, KeyArray],
                            batch: Minibatch) -> Tuple[TrainState, Metric]:
        state, rng = runner_state
        inputs, labels = batch

        rng, dropout_rng = jax.random.split(rng)

        def loss_fn(params):
          logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            inputs,
            mutable=["batch_stats"],
            rngs={"dropout": dropout_rng},
          )
          onehot = jax.nn.one_hot(labels, config.n_classes, dtype=config.dtype)
          loss = optax.softmax_cross_entropy(logits, onehot).mean()
          accuracy = (logits.argmax(axis=-1) == labels).mean()

          return loss, (new_model_state, accuracy)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux_vals), grad = grad_fn(state.params)
        new_model_state, acc = aux_vals

        new_state = state.apply_gradients(
          grads=grad,
          batch_stats=new_model_state["batch_stats"])
        train_metrics = {"loss": loss, "accuracy": acc}

        def evaluate(params):
          logits= new_state.apply_fn(
            {"params": params, "batch_stats": new_state.batch_stats},
            val_inputs, False,
            mutable=False,
          )
          onehot = jax.nn.one_hot(val_labels, config.n_classes,
                                  dtype=config.dtype)
          loss = optax.softmax_cross_entropy(logits, onehot).mean()
          accuracy = (logits.argmax(axis=-1) == val_labels).mean()

          return loss, accuracy
        
        val_loss, val_acc = evaluate(new_state.params)
        val_metrics = {"loss": val_loss, "accuracy": val_acc}
        
        return (new_state, rng), {"train": train_metrics, "val": val_metrics}
      
      rng, _rng = jax.random.split(rng)
      indices = jax.random.permutation(_rng, train_set_size)
      indices = indices[:n_minibatches * minibatch_size]

      batch = train_inputs[indices], train_labels[indices]
      minibatches = jax.tree_map(
        lambda x: jnp.reshape(x, (n_minibatches, -1) + x.shape[1:]),
        batch)
      
      train_state, metrics = jax.lax.scan(_update_minibatch,
                                          (train_state, rng),
                                          minibatches)
      
      return train_state, metrics
    
    runner_state = state, rng
    runner_state, metrics = jax.lax.scan(_train_one_epoch,
                                         runner_state,
                                         None,
                                         config.n_epochs)
    
    return {"train_state": runner_state[0],
            "metrics": metrics,
            "rng": runner_state[1]}
  
  return train


def train_vae(rng: KeyArray, config: TrainConfig, train_ds, val_ds):
  out_dir = os.path.join("outputs", config.run_name)
  os.makedirs(out_dir, exist_ok=True)

  model = VAE(config.model)
  rng, init_rng, dropout_rng, reparam_rng = jax.random.split(rng, 4)
  init_inp = jnp.ones((1, 250, 250, 3), dtype=config.dtype)
  params = model.init({"params": init_rng,
                       "dropout": dropout_rng},
                      reparam_rng, init_inp)["params"]
  tx = optax.chain(optax.clip_by_global_norm(config.clip_norm),
                   optax.adam(config.lr))
  state = TrainState.create(apply_fn=model.apply,
                            params=params,
                            tx=tx,
                            batch_stats=None)
  
  rng, _rng = jax.random.split(rng)
  z = jax.random.normal(_rng, (64, config.model.latent_dim))
  best_loss = float(np.inf)

  for i_epoch in range(config.n_epochs):
    epoch_metrics = {"loss": [], "bce_loss": [], "kld_loss": []}
    for minibatch in train_ds:
      (state, rng), metrics = vae_train_step((state, rng), jnp.array(minibatch, dtype=config.dtype))
      
      for k in epoch_metrics.keys():
        epoch_metrics[k].append(metrics[k].item())
    
    for k in epoch_metrics.keys():
      epoch_metrics[k] = np.mean(epoch_metrics[k])
    
    rng, _rng = jax.random.split(rng)
    metrics, comparison, sample = vae_eval(_rng, state, val_ds, z)

    vae_utils.save_image(comparison, f"{out_dir}/reconstruction_{i_epoch}.png", nrow=8)
    vae_utils.save_image(sample, f"{out_dir}/sample_{i_epoch}.png", nrow=8)

    print(f"| Epoch {i_epoch} | train | loss: {epoch_metrics['loss']:.3f} "
          f"| bce_loss: {epoch_metrics['bce_loss']:.3f} "
          f"| kld_loss: {epoch_metrics['kld_loss']:.3f}\n"
          f"| Epoch {i_epoch} |  val  | loss: {metrics['loss']:.3f} "
          f"| bce_loss: {metrics['bce_loss']:.3f} "
          f"| kld_loss: {metrics['kld_loss']:.3f}\n")
    
    if metrics["loss"] < best_loss:
      best_loss = metrics["loss"]
      save_checkpoint(state, out_dir)
  

def save_checkpoint(state: TrainState, out_dir: str) -> None:
  step = int(state.step)
  checkpoints.save_checkpoint(out_dir, state, step)
  logging.info(f"Saved checkpoint step {step}")