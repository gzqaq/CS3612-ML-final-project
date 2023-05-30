from train import make_train, save_checkpoint
from dataset import get_data
from configs import parse_user_flags

import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("batch_size", 256, "Minibatch size")
flags.DEFINE_float("clip_norm", 1.01, "Clip grad norm")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("n_classes", 10, "Number of classes")
flags.DEFINE_string("dtype", "float32", "Data type")
flags.DEFINE_string("model_config", "task_1", "Name of model config")
flags.DEFINE_string("run_name", "debug", "Name of this run")

def save_train_result(res):
  save_pth = os.path.join("outputs", FLAGS.run_name)
  os.makedirs(save_pth, exist_ok=True)

  save_checkpoint(res["train_state"], save_pth)
  
  metrics = jax.tree_map(np.array, res["metrics"])
  with open(os.path.join(save_pth, "metrics.pkl"), "wb") as fd:
    pickle.dump(metrics, fd)
  logging.info("Saved metrics to %s", os.path.join(save_pth, "metrics.pkl"))

def main(_):
  config = parse_user_flags(FLAGS)

  X_train, X_test, y_train, y_test = get_data("./dataset")
  train_imgs = jnp.einsum("bchw->bhwc", jnp.array(X_train, config.dtype))
  train_labels = jnp.array(y_train, dtype=config.dtype)
  test_imgs = jnp.einsum("bchw->bhwc", jnp.array(X_test, config.dtype))
  test_labels = jnp.array(y_test, dtype=config.dtype)

  train = make_train(config,
                     (jnp.array(train_imgs), jnp.array(train_labels)),
                     (jnp.array(test_imgs), jnp.array(test_labels)))
  
  rng = jax.random.PRNGKey(config.seed)
  res = jax.jit(train)(rng)

  save_train_result(res)


if __name__ == "__main__":
  app.run(main)