import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp

import jax_cfd.base as cfd
from .spectral import time_stepping,equations

from typing import Callable



def generate_trajectory_fn_FPC(
    Re: float,
    fforc: float,
    mask: jnp.ndarray,
    eta: float,
    delta: float,
    dt_stable: float,
    grid: cfd.grids.Grid,
    t_substep: float=1.,
    n_substep: float=1.,
    smooth: bool=True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    

    step_fn = time_stepping.crank_nicolson_rk2(
    equations.NS2D_ConstForc_VPM(Re,grid,fforc,smooth,mask,eta,delta), dt_stable)

    sub_step_fn = jax.jit(cfd.funcutils.repeated(jax.remat(step_fn), t_substep))

    trajectory_fn = jax.jit(cfd.funcutils.trajectory(jax.remat(sub_step_fn), n_substep, start_with_input=True))
    
    return trajectory_fn