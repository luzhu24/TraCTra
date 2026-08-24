""" To do: convert to jax-based data augmentation """
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

from typing import Union
Array = Union[np.ndarray, jnp.ndarray]

def translate_x(omega: Array) -> Array:
  Nx, _ = omega.shape[:2]
  shift = np.random.randint(0, Nx)
  omega_translated = np.roll(omega, shift, axis=0)
  return omega_translated

def shift_reflect_y(omega: Array) -> Array:
  _, Ny = omega.shape[:2]
  shift = np.random.randint(0, 8)

  for j in range(shift):
    omega = -np.flip(omega, axis=1)
  
  shift *= Ny // 8
  omega_shift_ref = np.roll(omega, shift, axis=1)
  return omega_shift_ref