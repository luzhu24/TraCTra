"""3D pseudospectral equations following Kim, Moin and Moser (JFM, 1987)."""

import dataclasses
from typing import Callable, Tuple, Optional

import jax as jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types as spectral_types
from jax_cfd.spectral import utils as spectral_utils

#===============================================================================
def KMM_velvort_to_velocity(
    grid: grids.Grid
) -> Callable[[spectral_types.Array,spectral_types.Array,
               spectral_types.Array,spectral_types.Array,
               spectral_types.Array],
               Tuple[spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array]]:
  """Constructs a function that recomputes the three velocity components
  vx, vy, vz and the two missing vorticity components 'omx' and 'omz'
  from the Laplacian Lapl(vy) and the associated vorticity 'omy'
  in the spirit of Kim, Moin & Moser [1]. All this is done in Fourier space.
  To do so, we first solve a Poisson equation to obtain vy from Lapl(vy).
  Next, we consider the definition of 'omy' and the continuity equation to
  recompute also the remaining velocity components (i.e. their Fourier coeff.).

  Args:
    grid: the grid underlying the vorticity field.

  Returns:
    A function that takes the Laplacian of vy and the vorticity omy (rfftn)
    and returns a (3D) velocity vector field.

  Reference:
    [1] J. Kim, P. Moin, R. Moser, "Turbulence statistics in fully developed
        channel flow at low Reynolds number", J. Fluid Mech., Volume 177,
        1987, Pages 133-166, https://doi.org/10.1017/S0022112087000892.
  """

  kx, ky, kz = grid.rfft_mesh()
  laplace = (2j * jnp.pi)** 2 * (kx**2 + ky**2 + kz**2)
  laplace = laplace.at[0, 0, 0].set(1) # essential, otherwise we catch a NaN or inf below at 1/laplace
                                       # -> when jitting, this leads to grad=NaN everywhere ...
  prefac  = 1j/(2. * jnp.pi * (kx**2 + kz**2))
  prefac  = prefac.at[0, :, 0].set(1j) # essential, otherwise we catch a NaN or inf here
                                       # -> when jitting, this leads to grad=NaN everywhere ...

  # pytype: disable=attribute-error  # jnp-type

  def ret(laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat):
    vyhat = 1 / laplace * laplvy_hat
    vyhat = vyhat.at[0, :, 0].set(jnp.squeeze(vy00_hat)) # set kx=kz=0-modes
    #vyhat = vyhat.at[0, 0, 0].set(0j)                # set kx=ky=kz=0 to 0

    grady_vy_hat  = 2j * jnp.pi * ky * vyhat
    grady_omy_hat = 2j * jnp.pi * ky * omy_hat

    vxhat = prefac * (kx * grady_vy_hat - kz * omy_hat)
    vzhat = prefac * (kz * grady_vy_hat + kx * omy_hat)
    vxhat = vxhat.at[0, :, 0].set(jnp.squeeze(vx00_hat)) # set kx=kz=0-modes
    vzhat = vzhat.at[0, :, 0].set(jnp.squeeze(vz00_hat)) # set kx=kz=0-modes
    #vxhat = vxhat.at[0, 0, 0].set(0j)        # set kx=ky=kz=0-mode to 0
    #vzhat = vzhat.at[0, 0, 0].set(0j)        # set kx=ky=kz=0-mode to 0

    omx_hat =  2j * jnp.pi * (ky * vzhat - kz * vyhat)
    omz_hat =  2j * jnp.pi * (kx * vyhat - ky * vxhat)

    return vxhat, vyhat, vzhat, omx_hat, omz_hat

  return ret

def KMM_velocity_to_velvort(
    grid: grids.Grid
) -> Callable[[spectral_types.Array,spectral_types.Array,
               spectral_types.Array],
               Tuple[spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array,
                     spectral_types.Array]]:
  """Constructs a function that computes the fields required to integrate
  a flow state forward in time following the Lapl(vy)-omy-formulation of
  Kim, Moin & Moser [1]. Input paramters are vx, vy and vz in physical space.

  Args:
    grid: the grid underlying the vorticity field.

  Returns:
    A function that takes the three velocity components of a given flow state
    and returns the Laplacian Lapl(vy), the associated vorticity 'omy' as well
    as the kx=kz=0 modes of the Fourier transforms of vx, vy and vz.

  Reference:
    [1] J. Kim, P. Moin, R. Moser, "Turbulence statistics in fully developed
        channel flow at low Reynolds number", J. Fluid Mech., Volume 177,
        1987, Pages 133-166, https://doi.org/10.1017/S0022112087000892.
  """

  kx, ky, kz = grid.rfft_mesh()
  laplace = (2j * jnp.pi)** 2 * (kx**2 + ky**2 + kz**2)
  #laplace = laplace.at[0, 0, 0].set(0.)

  def ret(vx, vy, vz):
    vx_hat = jnp.fft.rfftn(vx)
    vy_hat = jnp.fft.rfftn(vy)
    vz_hat = jnp.fft.rfftn(vz)
    #
    vx00_hat = vx_hat[0, :, 0] # kx=kz=0 modes for vx
    vy00_hat = vy_hat[0, :, 0] # kx=kz=0 modes for vy
    vz00_hat = vz_hat[0, :, 0] # kx=kz=0 modes for vz
    #
    laplvy_hat = laplace * vy_hat
    omy_hat = 2j * jnp.pi * (kz * vx_hat - kx * vz_hat)
    return laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat

  return ret


#===============================================================================
def brick_wall_filter_3d(grid: grids.Grid):
  """Implements the 2/3 rule."""
  # note: rfftn operates as rfft along the last axis (i.e. taking care of the
  #       symmetry for purely real data, whereas it treates all remaining axes
  #       as fftn, i.e. ignoring the fact that the negative wavenumbers do not
  #       carry additional information
  #       (https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html)
  #
  nx, ny, nz = grid.shape
  filter_ = jnp.zeros((nx, ny, nz // 2 + 1))
  # Notation: we think of the filter matrix of a cube centered at the
  # origin at (0, 0, 0) in the (kx,ky,kz) space. Since data is purely
  # positive along the last dimension (kz), it is actually a half-volume
  # -> in the following, we treat the following four quadrants separately:
  #  (I)   kx > 0, ky > 0, kz > 0
  #  (II)  kx < 0, ky > 0, kz > 0
  #  (III) kx < 0, ky < 0, kz > 0
  #  (IV)  kx > 0, ky < 0, kz > 0

  # xy-Quadrant I
  filter_ = filter_.at[:int(2 / 3 * nx) // 2,  \
                       :int(2 / 3 * ny) // 2,  \
                       :int(2 / 3 * (nz // 2 + 1))].set(1)
  # xy-Quadrant II
  filter_ = filter_.at[-int(2 / 3 * nx) // 2:, \
                       :int(2 / 3 * ny) // 2,  \
                       :int(2 / 3 * (nz // 2 + 1))].set(1)
  # xy-Quadrant III
  filter_ = filter_.at[-int(2 / 3 * nx) // 2:, \
                       -int(2 / 3 * ny) // 2:, \
                       :int(2 / 3 * (nz // 2 + 1))].set(1)
  # xy-Quadrant IV
  filter_ = filter_.at[:int(2 / 3 * nx) // 2,  \
                       -int(2 / 3 * ny) // 2:, \
                       :int(2 / 3 * (nz // 2 + 1))].set(1)
  return filter_
