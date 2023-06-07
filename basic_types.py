import jax
import numpy as np
from typing import Union, Tuple, Dict, Callable, Optional

Array = Union[jax.Array, np.ndarray]
KeyArray = Union[jax.Array, jax._src.prng.PRNGKeyArray]

Minibatch = Tuple[jax.Array, jax.Array]

Metric = Dict[str, float]
