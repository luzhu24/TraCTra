import jax
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
from spectral3d import utils

from typing import Optional, Sequence
from jax import Array


jax.config.update("jax_enable_x64", False) #--LZ


# function to generate on-attractor random initial states 
def set_initField(
    grid: cfd.grids.Grid,
    init_from_lam: bool=False,
    base: Optional[Sequence[Array]] = None,
    max_vel: float=1.,
    seed: float=None):
  #
  if seed is None: 
    seed = np.random.randint(1e4)

  v_init = cfd.initial_conditions.filtered_velocity_field(
                                   jax.random.PRNGKey(seed),\
                                   grid=grid,\
                                   maximum_velocity=max_vel,
                                   iterations=5)

  velvort_prepare = utils.KMM_velocity_to_velvort(grid)

  laplvy_init_rft,\
  omy_init_rft,\
  vx00_init_rft,\
  vy00_init_rft,\
  vz00_init_rft = velvort_prepare(base[0]+v_init[0].data, base[1]+v_init[1].data, base[2]+v_init[2].data)
  del v_init

  xvec_init_rft = (laplvy_init_rft, omy_init_rft, \
                   vx00_init_rft, vy00_init_rft, vz00_init_rft)
  del laplvy_init_rft, omy_init_rft, vx00_init_rft, vy00_init_rft, vz00_init_rft

  return xvec_init_rft


