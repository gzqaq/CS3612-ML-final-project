from basic_types import KeyArray, Array, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def PCA(data: Array, k: int = 2) -> Array:
  data_centered = data - jnp.mean(data, axis=0)
  cov_mat = jnp.cov(data_centered, rowvar=False)
  eig_vals, eig_vecs = jnp.linalg.eigh(cov_mat)

  top_k = eig_vecs[eig_vals.argsort()][-k:]
  projected_data = jnp.dot(data_centered, top_k.T)

  return projected_data


def tsne(
    rng: Optional[KeyArray],
    data: Array,
    k: int = 2,
    perplexity: int = 30,
    lr: float = 10,
    n_iter: int = 1000,
) -> Array:
  @jax.jit
  def _fill_diagonal(arr, val):
    i, j = jnp.diag_indices(min(arr.shape[-2:]))
    return arr.at[..., i, j].set(val)

  @jax.jit
  def softmax(inp):
    exp = jnp.exp(inp - inp.max(axis=1).reshape(-1, 1))
    exp = _fill_diagonal(exp, 0.0) + 1e-8

    return exp / exp.sum(axis=1).reshape(-1, 1)

  @jax.jit
  def neg_squared_euc_dist(data):
    sum_X = jnp.sum(jnp.square(data), axis=1)
    return -jnp.add(jnp.add(-2 * jnp.dot(data, data.T), sum_X).T, sum_X)

  @jax.jit
  def compute_prob_mat(dist, sigmas):
    return softmax(dist / (2 * jnp.square(sigmas.reshape(-1, 1))))

  @jax.jit
  def compute_perplexity(dist, sigmas):
    prob_mat = compute_prob_mat(dist, sigmas)
    entropy = -jnp.sum(prob_mat * jnp.log(prob_mat), 1)

    return jnp.exp(entropy)

  def search_optimal_sigmas(dist, perplexity, max_iter=1000):
    eval_fn = lambda sigma: compute_perplexity(dist, sigma)
    search_range = jnp.repeat(jnp.array([[0.0, 1000.0]]), dist.shape[0], 0)

    def _search_step(runner_state, _):
      rg = runner_state
      guess = jnp.mean(rg, axis=1)
      val = eval_fn(guess)
      pred = val > perplexity

      true_rg = rg.at[:, 1].set(guess)
      false_rg = rg.at[:, 0].set(guess)

      return jnp.where(pred[:, None], true_rg, false_rg), guess

    _, guesses = jax.lax.scan(_search_step, search_range, None, max_iter)
    return guesses[-1]

  # Compute P
  dist = neg_squared_euc_dist(data)
  sigmas = search_optimal_sigmas(dist, perplexity)
  p_cond = compute_prob_mat(dist, sigmas)
  P = (p_cond + p_cond.T) / (2.0 * p_cond.shape[0])

  @jax.jit
  def q_tsne(data):
    dist = neg_squared_euc_dist(data)
    inv_dist = jnp.power(1.0 - dist, -1)
    inv_dist = _fill_diagonal(inv_dist, 0.0)

    return inv_dist / jnp.sum(inv_dist), inv_dist

  @jax.jit
  def tsne_grad(Q, lower_repr, inv_dist):
    pq_diff = jnp.expand_dims(P - Q, 2)
    repr_diff = lower_repr[:, None, ...] - lower_repr[None, ...]
    repr_diff_wt = repr_diff * inv_dist[:, :, None]

    return 4.0 * (pq_diff * repr_diff_wt).sum(axis=1)

  if rng is None:
    low_repr = PCA(data, k)
  else:
    low_repr = jax.random.uniform(rng, (data.shape[0], k)) / jnp.sqrt(k)

  tx = optax.sgd(lr, momentum=0.9)
  state = TrainState.create(params=low_repr, tx=tx, apply_fn=None)

  def _update_step(runner_state, _):
    state = runner_state
    Q, dist = q_tsne(state.params)
    grads = tsne_grad(Q, state.params, dist)

    return state.apply_gradients(grads=grads), None

  state, _ = jax.lax.scan(_update_step, state, None, n_iter)

  return state.params
