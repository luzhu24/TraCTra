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

"""Forcing functions for 3D Navier-Stokes equations."""

# TODO(jamieas): change the signature for all forcing functions so that they
# close over `grid`.

import functools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import filter_utils
from jax_cfd.base import grids
from jax_cfd.base import validation_problems

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


def taylor_green_forcing(
    grid: grids.Grid,
    scale: float = 1,
    kf: int = 2,
    forcedir: int = 0,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None
) -> ForcingFn:
  """Constant driving forced in the form of Taylor-Green vorcities.
     The axis (along with the forcing is acting) can be freely chosen.
     Force-dir means in this case that the associated vorticity is
     unidirectional, e.g.: forcedir=0 (x-dir) -> f(y,z) with omx > 0.
     Note that for now, the forcing wavenumber is the same in both
     direction. """
  # Put force on same offset, grid as velocity components
  if grid.ndim == 2:
    u, v = validation_problems.TaylorGreen(shape=grid.shape[:2], kx=kf, ky=kf).velocity()
    u = grids.GridArray(u.data * scale, offsets[0], grid)
    v = grids.GridArray(v.data * scale, offsets[1], grid)
    f = (u, v)

  elif grid.ndim == 3:
    if forcedir == 0:
      # append z-dimension to u,v arrays
      v, w = validation_problems.TaylorGreen(shape=grid.shape[1:], kx=kf, ky=kf).velocity()
      v_data = jnp.broadcast_to(jnp.expand_dims(v.data * scale, 0), grid.shape)
      w_data = jnp.broadcast_to(jnp.expand_dims(w.data * scale, 0), grid.shape)
      v = grids.GridArray(v_data, offsets[1], grid)
      w = grids.GridArray(w_data, offsets[2], grid)
      u = grids.GridArray(jnp.zeros_like(v.data), offsets[0], grid)
      f = (u, v, w)
    elif forcedir == 1:
      print("Check for this forcing the order: (u,w) or (w,u)?")
      # append y-dimension to u,w arrays
      w, u = validation_problems.TaylorGreen(shape=(grid.shape[0],grid.shape[2]), kx=kf, ky=kf).velocity()
      u_data = jnp.broadcast_to(jnp.expand_dims(u.data * scale, 1), grid.shape)
      w_data = jnp.broadcast_to(jnp.expand_dims(w.data * scale, 1), grid.shape)
      u = grids.GridArray(u_data, offsets[0], grid)
      w = grids.GridArray(w_data, offsets[2], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), offsets[1], grid)
      f = (u, v, w)
    elif forcedir == 2:
      # append z-dimension to u,v arrays
      u, v = validation_problems.TaylorGreen(shape=grid.shape[:2], kx=kf, ky=kf).velocity()
      u_data = jnp.broadcast_to(jnp.expand_dims(u.data * scale, -1), grid.shape)
      v_data = jnp.broadcast_to(jnp.expand_dims(v.data * scale, -1), grid.shape)
      u = grids.GridArray(u_data, offsets[0], grid)
      v = grids.GridArray(v_data, offsets[1], grid)
      w = grids.GridArray(jnp.zeros_like(u.data), offsets[2], grid)
      f = (u, v, w)
  else:
    raise NotImplementedError

  def forcing(v):
    del v
    return f
  return forcing


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    kf: int = 2,
    forcedir: int = 0,
    forcing2d: bool = False,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  """Returns the Kolmogorov forcing function for turbulence in 2D/3D.
     This can be either 1D (i.e. the forcing depends of a single spatial dir.),
     or 2D in that the forcing depends on the two cross-stream directions,
     as used in Lucas & Kerswell (JFM, 2017). The axis (along with the forcing
     is acting) can be freely chosen. In the 1D-forcing, the forcing oscillates
     along the 'next' dimension, e.g. fx(y), fy(z), ... """
  # note i have change the forcing to cos(kf*y)
  if offsets is None:
    offsets = grid.cell_faces

  if grid.ndim == 2:
    if forcing2d:
      raise NotImplementedError

    if forcedir == 0:
      y = grid.mesh(offsets[0])[1]
      u = scale * grids.GridArray(jnp.cos(kf * y), offsets[0], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), offsets[1], grid)
    elif forcedir == 1:
      x = grid.mesh(offsets[1])[0]
      v = scale * grids.GridArray(jnp.cos(kf * x), offsets[1], grid)
      u = grids.GridArray(jnp.zeros_like(v.data), offsets[0], grid)
    else:
      raise NotImplementedError

    f = (u, v)

  elif grid.ndim == 3 and forcing2d == False:
    if forcedir == 0:
      y = grid.mesh(offsets[0])[1]
      u = scale * grids.GridArray(jnp.cos(kf * y), offsets[0], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), offsets[1], grid)
      w = grids.GridArray(jnp.zeros_like(u.data), offsets[2], grid)
    elif forcedir == 1:
      z = grid.mesh(offsets[1])[2]
      v = scale * grids.GridArray(jnp.cos(kf * z), offsets[1], grid)
      u = grids.GridArray(jnp.zeros_like(v.data), offsets[0], grid)
      w = grids.GridArray(jnp.zeros_like(v.data), offsets[2], grid)
    elif forcedir == 2:
      x = grid.mesh(offsets[2])[0]
      w = scale * grids.GridArray(jnp.cos(kf * x), offsets[2], grid)
      u = grids.GridArray(jnp.zeros_like(w.data), offsets[0], grid)
      v = grids.GridArray(jnp.zeros_like(w.data), offsets[1], grid)
    else:
      raise NotImplementedError

    f = (u, v, w)

  elif grid.ndim == 3 and forcing2d == True:
    if forcedir == 0:
      y = grid.mesh(offsets[0])[1]
      z = grid.mesh(offsets[0])[2]
      u = scale * grids.GridArray(jnp.cos(kf * y) * jnp.cos(kf * z), offsets[0], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), offsets[1], grid)
      w = grids.GridArray(jnp.zeros_like(u.data), offsets[2], grid)
    elif forcedir == 1:
      x = grid.mesh(offsets[1])[0]
      z = grid.mesh(offsets[1])[2]
      v = scale * grids.GridArray(jnp.cos(kf * z) * jnp.cos(kf * x), offsets[1], grid)
      u = grids.GridArray(jnp.zeros_like(v.data), offsets[0], grid)
      w = grids.GridArray(jnp.zeros_like(v.data), offsets[2], grid)
    elif forcedir == 2:
      x = grid.mesh(offsets[2])[0]
      y = grid.mesh(offsets[2])[1]
      w = scale * grids.GridArray(jnp.cos(kf * x)* jnp.cos(kf * y), offsets[2], grid)
      u = grids.GridArray(jnp.zeros_like(w.data), offsets[0], grid)
      v = grids.GridArray(jnp.zeros_like(w.data), offsets[1], grid)
    else:
      raise NotImplementedError

    f = (u, v, w)

  def forcing(v):
    del v
    return f
  return forcing


def nonhelical_sinusoidal_forcing(
    grid: grids.Grid,
    scale: float = 1,
    kf: int = 2,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  """ Returns a three-dimensional sinusoidal forcing that is strictly nonhelical
     (cf. e.g. Linkmann et al., PRF, 2017). Forcing is applied to wavenumbers
     in the range k \in [0,kf] only. """

  if grid.ndim == 2:
    raise NotImplementedError
  elif grid.ndim == 3:
    Nx, Ny, Nz = grid.shape
    #xu = grid.mesh(offsets[0])[0]
    yu = grid.mesh(offsets[0])[1]
    zu = grid.mesh(offsets[0])[2]
    xv = grid.mesh(offsets[1])[0]
    #yv = grid.mesh(offsets[1])[1]
    zv = grid.mesh(offsets[1])[2]
    xw = grid.mesh(offsets[2])[0]
    yw = grid.mesh(offsets[2])[1]
    #zw = grid.mesh(offsets[2])[2]
    #
    uf = jnp.zeros((grid.shape))
    vf = jnp.zeros((grid.shape))
    wf = jnp.zeros((grid.shape))
    #
    ### for k in range(kfb,kfe+1):
    for k in range(1,kf+1):
      uf += jnp.sin(k * zu) + jnp.sin(k * yu)
      vf += jnp.sin(k * xv) + jnp.sin(k * zv)
      wf += jnp.sin(k * yw) + jnp.sin(k * xw)

    u = scale * grids.GridArray(uf, offsets[0], grid)
    v = scale * grids.GridArray(vf, offsets[0], grid)
    w = scale * grids.GridArray(wf, offsets[0], grid)

  else:
    raise NotImplementedError

  f = (u, v, w)

  def forcing(v):
    del v
    return f
  return forcing




def linear_forcing(grid, coefficient: float) -> ForcingFn:
  """Linear forcing, proportional to velocity."""
  del grid

  def forcing(v):
    return tuple(coefficient * u.array for u in v)
  return forcing


def no_forcing(grid):
  """Zero-valued forcing field for unforced simulations."""
  del grid

  def forcing(v):
    return tuple(0 * u.array for u in v)
  return forcing


def sum_forcings(*forcings: ForcingFn) -> ForcingFn:
  """Sum multiple forcing functions."""
  def forcing(v):
    return equations.sum_fields(*[forcing(v) for forcing in forcings])
  return forcing


FORCING_FUNCTIONS = dict(kolmogorov=kolmogorov_forcing,
                         taylor_green=taylor_green_forcing)


def simple_turbulence_forcing(
    grid: grids.Grid,
    constant_magnitude: float = 0,
    constant_wavenumber: int = 2,
    linear_coefficient: float = 0,
    forcing_type: str = 'kolmogorov',
) -> ForcingFn:
  """Returns a forcing function for turbulence in 2D or 3D.

  2D turbulence needs a driving force injecting energy at intermediate
  length-scales, and a damping force at long length-scales to avoid all energy
  accumulating in giant vorticies. This can be achieved with
  `constant_magnitude > 0` and `linear_coefficient < 0`.

  3D turbulence only needs a driving force at the longest length-scale (damping
  happens at the smallest length-scales due to viscosity and/or numerical
  dispersion). This can be achieved with `constant_magnitude = 0` and
  `linear_coefficient > 0`.

  Args:
    grid: grid on which to simulate.
    constant_magnitude: magnitude for constant forcing with Taylor-Green
      vortices.
    constant_wavenumber: wavenumber for constant forcing with Taylor-Green
      vortices.
    linear_coefficient: forcing coefficient proportional to velocity, for
      either driving or damping based on the sign.
    forcing_type: String that specifies forcing. This must specify the name of
      function declared in FORCING_FUNCTIONS (taylor_green, etc.)

  Returns:
    Forcing function.
  """

  linear_force = linear_forcing(grid, linear_coefficient)
  constant_force_fn = FORCING_FUNCTIONS.get(forcing_type)
  if constant_force_fn is None:
    raise ValueError('Unknown `forcing_type`. '
                     f'Expected one of {list(FORCING_FUNCTIONS.keys())}; '
                     f'got {forcing_type}.')
  constant_force = constant_force_fn(grid, constant_magnitude,
                                     constant_wavenumber)
  return sum_forcings(linear_force, constant_force)


def filtered_forcing(
    spectral_density: Callable[[Array], Array],
    grid: grids.Grid,
) -> ForcingFn:
  """Apply forcing as a function of angular frequency.

  Args:
    spectral_density: if `x_hat` is a Fourier component of the velocity with
      angular frequency `k` then the forcing applied to `x_hat` is
      `spectral_density(k)`.
    grid: object representing spatial discretization.
  Returns:
    A forcing function that applies filtered forcing.
  """
  def forcing(v):
    filter_ = grids.applied(
        functools.partial(filter_utils.filter, spectral_density, grid=grid))
    return tuple(filter_(u.array) for u in v)
  return forcing


def filtered_linear_forcing(
    lower_wavenumber: float,
    upper_wavenumber: float,
    coefficient: float,
    grid: grids.Grid,
) -> ForcingFn:
  """Apply linear forcing to low frequency components of the velocity field.

  Args:
    lower_wavenumber: the minimum wavenumber to which forcing should be
      applied.
    upper_wavenumber: the maximum wavenumber to which forcing should be
      applied.
    coefficient: the linear coefficient for forcing applied to components with
      wavenumber below `threshold`.
    grid: object representing spatial discretization.
  Returns:
    A forcing function that applies filtered linear forcing.
  """
  def spectral_density(k):
    return jnp.where(((k >= 2 * jnp.pi * lower_wavenumber) &
                      (k <= 2 * jnp.pi * upper_wavenumber)),
                     coefficient,
                     0)
  return filtered_forcing(spectral_density, grid)

