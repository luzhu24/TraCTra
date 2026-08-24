""" Routines for generating test datasets with jax-cfd. 
    Warning: Data may be transposed (in space) compared with legacy data
    from in-house spectral solver. """
from typing import Callable, Tuple, List, Union

import numpy as np
import jax
import jax.numpy as jnp

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
import forcings
import equations
import time_stepping

jax.config.update("jax_enable_x64", True)

Array = Union[np.array, jnp.array]

def periodic3d_step(
    grid: cfd.grids.Grid,
    visc: float,
    kf: int=4,
    forcing: str="Kolmogorov1D",
    forcedir: int=0,
    smooth=True,
) -> Callable[[Tuple[Array]], Tuple[Array]]:
  """ Function to march a timestep based on the spectral discretization """
  offsets = ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.))

   # pylint: disable=g-long-lambda

  if forcing=="TaylorGreen": # TaylorGreen forcing
    forcing2d = True
    forcing_fn = lambda grid: forcings.taylor_green_forcing(
                             grid,
                             kf=kf,
                             forcedir=forcedir,
                             offsets=offsets)
  elif forcing=="Kolmogorov1D": # Kolmogorov forcing à la (fx(y),0,0)
    forcing2d = False
    forcing_fn = lambda grid: forcings.kolmogorov_forcing(
                             grid,
                             kf=kf,
                             forcedir=forcedir,
                             forcing2d=forcing2d,
                             offsets=offsets
                             )
  elif forcing=="Kolmogorov2D": # Kolmogorov forcing à la (fx(y,z),0,0)
    forcing2d = True
    forcing_fn = lambda grid: forcings.kolmogorov_forcing(
                             grid,
                             kf=kf,
                             forcedir=forcedir,
                             forcing2d=forcing2d,
                             offsets=offsets
                             )

  elif forcing=="NonhelicalSinusoidal": # non-helical sinusoidal forcing in x,y,z
    forcing_fn = lambda grid: forcings.nonhelical_sinusoidal_forcing(
                             grid,
                             kf=kf,
                             offsets=offsets
                             )

  elif forcing=="Poiseuille1D": # --LZ
    forcing_fn = lambda grid: forcings.poiseuille_forcing(
                             grid,
                             kf=kf,
                             offsets=offsets
                             )

  return equations.NavierStokes3D(
             visc,
             grid,
             drag=0.0,
             smooth=smooth,
             forcing_fn=forcing_fn)

class SimulationConfig:
  def __init__(
    self, 
    grid: cfd.grids.Grid,
    visc: float, 
    kf: int=4,
    forcing: str="Kolmogorov1D",
    forcedir: int=0,
) -> Callable[[Tuple[Array]], Tuple[Array]]:
    """ Class framework for a triply-periodic DNS """
    self.grid = grid
    self.visc = visc
    self.Re = 1./self.visc
    self.kf = kf
    self.forcing = forcing
    self.forcedir = forcedir 
    self.ns_eqn_step = periodic3d_step(
                       grid=self.grid,
                       visc=self.visc,
                       kf=self.kf,
                       forcing=forcing,
                       forcedir=forcedir,
                       smooth=True)
    
    # laminar velocity to determine the required timestep
    #velocity_est = self.Re / kf ** 2
    velocity_est = 5.
    #dxmin = jnp.asarray(grid.step).min()
    dxmin = jnp.asarray(grid.step)[0]# --LZ
    cflTarget = 0.25
    self.dt_stable = (cflTarget * dxmin / velocity_est) 
    print("Choose velocity_est as ", velocity_est)
    print("Chosen time step: dt (CFL-target) =", self.dt_stable, "(", cflTarget, ")")
  
  def generate_time_forward_map(
      self,
      Nt: int,
      dt: float = None
  ) -> Callable[[Array], Array]:
    if dt is None:
      dt = self.dt_stable
    step_fn = time_stepping.crank_nicolson_rk4(self.ns_eqn_step, dt, screenout=True)
    time_forward_map = cfd.funcutils.repeated(jax.remat(step_fn), steps=Nt)
    return jax.jit(time_forward_map)

  def generate_trajectory_fn(
      self,
      T: float,
      t_substep: float=1.,
      dt: float=None
      ) -> Callable[[Tuple[cfd.grids.GridVariable]], Tuple[cfd.grids.GridVariable]]:
    if dt is None:
      N_steps_total = jnp.floor(T / self.dt_stable)
    else:
      N_steps_total = jnp.floor(T / dt)
    dt_exact = T / N_steps_total

    if t_substep < dt_exact:
      print("traj: dt_substep < dt_exact -> set dt_substep = dt_exact")
      t_substep = dt_exact
    #
    N_steps_per_substep = jnp.floor(t_substep / dt_exact)
    N_substeps = N_steps_total // N_steps_per_substep
    self.dt_exact = dt_exact
    self.N_steps_per_substep = N_steps_per_substep
    self.N_substeps = N_substeps
    print("DEBUG dt_exact, N_substeps, Nsteps_total:", dt_exact, N_substeps, N_steps_total)
    sub_step_fn = self.generate_time_forward_map(N_steps_per_substep, dt_exact)

    trajectory_fn = jax.jit(cfd.funcutils.trajectory(jax.remat(sub_step_fn), N_substeps))
    return trajectory_fn
