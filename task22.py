from train import train_vae
from task22_dataset import load_image, get_image_path
from configs import parse_user_flags

import jax
import numpy as np
import os
from absl import app, flags

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("batch_size", 256, "Minibatch size")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("n_classes", 10, "Useless")
flags.DEFINE_float("clip_norm", 1.01, "Clip grad norm")
flags.DEFINE_float("lr", 0.00001, "Learning rate")
flags.DEFINE_string("dtype", "bfloat16", "Data type")
flags.DEFINE_string("model_config", "task_22", "Name of model config")
flags.DEFINE_string("run_name", "debug", "Name of this run")


def main(_):
  config = parse_user_flags(FLAGS)
  np.random.seed(config.seed)
  rng = jax.random.PRNGKey(config.seed)

  ds = load_image(get_image_path("task22_dataset"))
  ds = np.einsum("bchw->bhwc", np.array(ds, dtype=np.float32))
  np.random.shuffle(ds)

  train_size = int(ds.shape[0] * 0.9) // config.batch_size * config.batch_size
  train_ds = ds[:train_size]
  val_ds = ds[train_size:]

  train_vae(rng, config, train_ds, val_ds)


if __name__ == "__main__":
  app.run(main)
