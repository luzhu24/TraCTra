# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pseudospectral equations."""

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax_cfd.base import boundaries
#from jax_cfd.base import forcings
from . import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils


TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]


@dataclasses.dataclass
class KuramotoSivashinsky(time_stepping.ImplicitExplicitODE):
  """Kuramoto–Sivashinsky (KS) equation split in implicit and explicit parts.

  The KS equation is
    u_t = - u_xx - u_xxxx - 1/2 * (u ** 2)_x

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    grid: underlying grid of the process
    smooth: smooth the non-linear term using the 3/2-rule
  """
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, = self.grid.rfft_axes()
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.linear_term = -self.two_pi_i_k ** 2 - self.two_pi_i_k ** 4
    self.rfft = spectral_utils.truncated_rfft if self.smooth else jnp.fft.rfft
    self.irfft = spectral_utils.padded_irfft if self.smooth else jnp.fft.irfft

  def explicit_terms(self, uhat):
    """Non-linear parts of the equation, namely `- 1/2 * (u ** 2)_x`."""
    uhat_squared = self.rfft(jnp.square(self.irfft(uhat)))
    return -0.5 * self.two_pi_i_k * uhat_squared

  def implicit_terms(self, uhat):
    """Linear parts of the equation, namely `- u_xx - u_xxxx`."""
    return self.linear_term * uhat

  def implicit_solve(self, uhat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    # TODO(dresdner) the same for all linear terms. generalize/refactor?
    return 1 / (1 - time_step * self.linear_term) * uhat


@dataclasses.dataclass
class ForcedBurgersEquation(time_stepping.ImplicitExplicitODE):
  """Burgers' Equation with the option to add a time-dependent forcing function."""
  viscosity: float
  grid: grids.Grid
  seed: int = 0
  forcing_module: Optional[
      ForcingModule] = spectral_forcings.random_forcing_module
  _forcing_fn = None

  def __post_init__(self):
    self.kx, = self.grid.rfft_axes()
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.linear_term = self.viscosity * self.two_pi_i_k ** 2
    self.rfft = spectral_utils.truncated_rfft
    self.irfft = spectral_utils.padded_irfft
    if self.forcing_module is None:
      self._forcing_fn = lambda t: jnp.zeros(1)
    else:
      self._forcing_fn = self.forcing_module(self.grid, self.seed)

  def explicit_terms(self, state):
    uhat, t = state
    dudx = self.two_pi_i_k * uhat

    f = self._forcing_fn(t)
    fhat = jnp.fft.rfft(f)

    advection = - self.rfft(self.irfft(uhat) * self.irfft(dudx))

    return (fhat + advection, 1.0)

  def implicit_terms(self, state):
    uhat, _ = state
    return (self.linear_term * uhat, 0.0)

  def implicit_solve(self, state, time_step):
    uhat, t = state
    return (1 / (1 - time_step * self.linear_term) * uhat, t)


def BurgersEquation(viscosity: float, grid: grids.Grid, seed: int = 0):
  """Standard, unforced Burgers' equation."""
  return ForcedBurgersEquation(
      viscosity=viscosity, grid=grid, seed=seed, forcing_module=None)


# pylint: disable=invalid-name
'''Take my raw array arr on this grid, assume its a cell-centered 2D periodic field (unless I override bc/offset), 
and turn it into a GridVariable that jax-cfd operators (grad, div, advect, etc.) can work with.'''
def _get_grid_variable(arr,
                       grid,
                       bc=boundaries.periodic_boundary_conditions(2),
                       offset=(0., 0.)):
  return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


@dataclasses.dataclass
class NavierStokes2D(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.
  """
  viscosity: float
  grid: grids.Grid
  drag: float = 0.
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  ibm_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  _forcing_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
    self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    self.linear_term = self.viscosity * self.laplace - self.drag

    # setup the forcing function with the caller-specified grid.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    if self.ibm_fn is not None:
      self._ibm_fn_with_grid = self.ibm_fn(self.grid)


  def explicit_terms(self, vorticity_hat):
    velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

    grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
    grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * vx + grad_y * vy)
    advection_hat = jnp.fft.rfftn(advection)

    if self.smooth is not None:
      advection_hat *= self.filter_

    terms = advection_hat

    if self.forcing_fn is not None:
      fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                           _get_grid_variable(vy, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
      terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                               (fx_hat, fy_hat))

    if self.ibm_fn is not None:
      fx, fy = self._ibm_fn_with_grid((_get_grid_variable(vx, self.grid),
                                       _get_grid_variable(vy, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(-fx.data), jnp.fft.rfft2(-fy.data)
      terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                               (fx_hat, fy_hat))

    return terms

  def implicit_terms(self, vorticity_hat):
    return self.linear_term * vorticity_hat

  def implicit_solve(self, vorticity_hat, time_step):
    return 1 / (1 - time_step * self.linear_term) * vorticity_hat


# pylint: disable=g-doc-args,g-doc-return-or-yield,invalid-name
def ForcedNavierStokes2D(viscosity, grid, smooth):
  """Sets up the flow that is used in Kochkov et al. [1].

  The authors of [1] based their work on Boffetta et al. [2].

  References:
    [1] Machine learning–accelerated computational fluid dynamics. Dmitrii
    Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
    Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
    e2101784118; DOI: 10.1073/pnas.2101784118.
    https://doi.org/10.1073/pnas.2101784118

    [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
    Annual review of fluid mechanics 44 (2012): 427-451.
    https://doi.org/10.1146/annurev-fluid-120710-101240
  """
  wave_number = 4
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets)
  return NavierStokes2D(
      viscosity,
      grid,
      drag=0.1,
      smooth=smooth,
      forcing_fn=forcing_fn)







@dataclasses.dataclass
class NonlinearSchrodinger(time_stepping.ImplicitExplicitODE):
  """Nonlinear schrodinger equation split in implicit and explicit parts.

  The NLS equation is
    `psi_t = -i psi_xx/8 - i|psi|^2 psi/2`

  Attributes:
    grid: underlying grid of the process
    smooth: smooth the non-linear by upsampling 2x in fourier and truncating
  """
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, = self.grid.fft_axes()
    assert len(self.kx) % 2 == 0, "Odd grid sizes not supported, try N even"
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.fft = spectral_utils.truncated_fft_2x if self.smooth else jnp.fft.fft
    self.ifft = spectral_utils.padded_ifft_2x if self.smooth else jnp.fft.ifft

  def explicit_terms(self, psihat):
    """Non-linear part of the equation `-i|psi|^2 psi/2`."""
    psi = self.ifft(psihat)
    ipsi_cubed = 1j * psi * jnp.abs(psi)**2
    ipsi_cubed_hat = self.fft(ipsi_cubed)
    return -ipsi_cubed_hat / 2

  def implicit_terms(self, psihat):
    """The diffusion term `-i psi_xx/8` to be handled implicitly."""
    return -1j * psihat * self.two_pi_i_k**2 / 8

  def implicit_solve(self, psihat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    return psihat / (1 - time_step * (-1j * self.two_pi_i_k**2 / 8))
  


###########################################################################
####################### LZ's modification #################################

#### IBM #####
@dataclasses.dataclass
class NS2D_ConstForc(time_stepping.ImplicitExplicitODE):

  re: float
  grid: grids.Grid
  mean_forc_x: float = 0
  drag: float = 0.
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  ibm_vel_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  _forcing_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    self.dx,self.dy = (jnp.pi*2j) *self.kx, (jnp.pi*2j) *self.ky
    self.ddx,self.ddy,self.dxy = self.dx**2, self.dy**2, self.dx*self.dy
    self.laplace = self.ddx + self.ddy
    self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)


    self.linear_term = [1/self.re * self.laplace - self.drag, -self.drag, ]#

    # setup the forcing function.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    if self.ibm_vel_fn is not None:
      self._ibm_vel_fn_with_grid = self.ibm_vel_fn(self.grid)

  def explicit_terms(self, qs_hat):
    w_hat,U0 = qs_hat
    velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
    uhat, vhat = velocity_solve(w_hat)
    u,v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)
    utot = u+U0
                       
    # NS
    dwdx,dwdy = jnp.fft.irfftn(self.dx*w_hat), jnp.fft.irfftn(self.dy*w_hat)
    advection =  -(utot*dwdx + v*dwdy)
    advection_hat = jnp.fft.rfftn(advection) 
    dU0_dt = self.mean_forc_x

    if self.forcing_fn is not None:
      fx, fy = self._forcing_fn_with_grid((_get_grid_variable(utot, self.grid), _get_grid_variable(v, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
      advection_hat += spectral_utils.spectral_curl_2d((self.kx, self.ky), (fx_hat, fy_hat))
      # mean momentum forcing in x
      dU0_dt += jnp.mean(fx.data)

    if self.ibm_vel_fn is not None:
      fx, fy = self._ibm_vel_fn_with_grid((_get_grid_variable(utot, self.grid),
                                       _get_grid_variable(v, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(-fx.data), jnp.fft.rfft2(-fy.data)
      advection_hat += spectral_utils.spectral_curl_2d((self.kx, self.ky),(fx_hat, fy_hat))

      # mean momentum forcing in x
      dU0_dt += jnp.mean(-fx.data)


    if self.smooth is not None:
      advection_hat *= self.filter_
    
    #return explicit_terms
    return [advection_hat, dU0_dt]


  def implicit_terms(self, qs_hat):
    return [lin*q for lin,q in zip(self.linear_term,qs_hat)]

  def implicit_solve(self, qs_hat, time_step):
    return [1/(1-time_step*lin)*q  for lin,q in zip(self.linear_term,qs_hat)]



def NS2D_ConstForc_VPM(re, grid, fforc, smooth, sdf, eta, delta):
  mean_forc_x = fforc/re
  offsets = ((0, 0), (0, 0))
  ibm_vel_fn = lambda grid: forcings.vpm_Dirichlet(
      grid, sdf=sdf, eta=eta, delta=delta, offsets=offsets)
  
  return NS2D_ConstForc(
      re, 
      grid,
      mean_forc_x=mean_forc_x,
      drag=0.,
      smooth=smooth,
      forcing_fn=None,
      ibm_vel_fn=ibm_vel_fn,)
