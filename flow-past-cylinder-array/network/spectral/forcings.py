"""Forcing functions for spectral equations."""
import functools
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax_cfd.base import grids

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]

def random_forcing_module(grid: grids.Grid,
                          seed: int = 0,
                          n: int = 20,
                          offset=(0,)):
  """Implements the forcing described in Bar-Sinai et al. [*].

  Args:
    grid: grid to use for the x-axis
    seed: random seed for computing the random waves
    n: number of random waves to use
    offset: offset for the x-axis. Defaults to (0,) for the Fourier basis.
  Returns:
    Time dependent forcing function.

  [*] Bar-Sinai, Yohai, Stephan Hoyer, Jason Hickey, and Michael P. Brenner.
  "Learning data-driven discretizations for partial differential equations."
  Proceedings of the National Academy of Sciences 116, no. 31 (2019):
  15344-15349.
  """

  key = jax.random.PRNGKey(seed)

  ks = jnp.array([3, 4, 5, 6])

  key, subkey = jax.random.split(key)
  kx = jax.random.choice(subkey, ks, shape=(n,))

  key, subkey = jax.random.split(key)
  amplitude = jax.random.uniform(subkey, minval=-0.5, maxval=0.5, shape=(n,))

  key, subkey = jax.random.split(key)
  omega = jax.random.uniform(subkey, minval=-0.4, maxval=0.4, shape=(n,))

  key, subkey = jax.random.split(key)
  phi = jax.random.uniform(subkey, minval=0, maxval=2 * jnp.pi, shape=(n,))

  xs, = grid.axes(offset=offset)

  def forcing_fn(t):

    @jnp.vectorize
    def eval_force(x):
      f = amplitude * jnp.sin(omega * t - x * kx + phi)
      return f.sum()

    return eval_force(xs)

  return forcing_fn


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    k: int = 2,
    swap_xy: bool = False,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  """Returns the Kolmogorov forcing function for turbulence in 2D."""
  if offsets is None:
    offsets = grid.cell_faces

  if swap_xy:
    x = grid.mesh(offsets[1])[0]
    v = scale * grids.GridArray(jnp.cos(k * x), offsets[1], grid)

    if grid.ndim == 2:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2), grid)
      f = (u, v)
    elif grid.ndim == 3:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError
  else:
    y = grid.mesh(offsets[0])[1]
    u = scale * grids.GridArray(jnp.cos(k * y), offsets[0], grid)

    if grid.ndim == 2:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1), grid)
      f = (u, v)
    elif grid.ndim == 3:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError

  def forcing(v):
    del v
    return f
  return forcing



def vpm_Dirichlet(  ##--LZ
    grid: grids.Grid,
    sdf: jnp.array, 
    eta: float, 
    delta: float, 
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  """Returns the volume penalty for noslip side wall.
     offsets depdend on the location of variables (e.g. u, v); for spectral scheme offsets is (0,0) for all variables"""
  if offsets is None:
    offsets = grid.cell_faces
  
  tfm1 = 1./eta
  mask = 0.5*(1+jnp.tanh(2./delta*sdf))

  def forcing(uv):
    return [grids.GridArray(tfm1*mask*u.data, u.offset, grid) for u in uv]
  return forcing



