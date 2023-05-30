from basic_types import Array, Minibatch, Metric, KeyArray
from configs import TrainConfig
from model import MyModel

import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import struct
from flax.training import train_state, checkpoints
from typing import Callable, Tuple, Any


class TrainState(train_state.TrainState):
  batch_stats: Any


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
  

def save_checkpoint(state: TrainState, out_dir: str) -> None:
  step = int(state.step)
  checkpoints.save_checkpoint(out_dir, state, step)
  logging.info(f"Saved checkpoint step {step}")