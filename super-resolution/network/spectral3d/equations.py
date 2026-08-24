"""3D pseudospectral equations following Kim, Moin and Moser (JFM, 1987)."""

import dataclasses
from typing import Callable, Tuple, Optional

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
#from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types as spectral_types
from jax_cfd.spectral import utils as spectral_utils
from . import utils as spectral_utils3d
from . import forcings

# pylint: disable=invalid-name
def _get_grid_variable(arr,
                       grid,
                       bc=boundaries.periodic_boundary_conditions(3),
                       offset=(0., 0., 0.)):
  return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


@dataclasses.dataclass
class NavierStokes3D(time_stepping.ImplicitExplicitODE):
  """Builds up a 3D Navier-Stokes solver based on the scheme
  outlined in Kim, Moin & Moser for doubly-periodic channel.
  Instead of the original primitive variable formulation, a 4th order equation
  for 'vy' and a 2nd order for the associated vorticity 'omy' are
  integrated in time, broken into implicit and explicit parts:

  |  d(q)/dt = L(q) + N(q)
  |          = 1/nu * Lapl(q) + N(q)
  |
  | with      q = (Lapl(v), omy)^T
  |      N_1(q) = -d/dy [d(H1)/dx + d(H3)/dz] + (d/dxx + d/dzz) H2
  |      N_2(q) = [d(H1)/dz - d(H3)/dx]
  |        H_i  = - u_j * d(u_i)/d_j          (non-linear term in NSE)

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms. Note: The kx=kz=0 is treated separately!

  Args:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.

  Reference:
    [1] J. Kim, P. Moin, R. Moser, "Turbulence statistics in fully developed
        channel flow at low Reynolds number", J. Fluid Mech., Volume 177,
        1987, Pages 133-166, https://doi.org/10.1017/S0022112087000892.
  """
  viscosity: float
  grid: grids.Grid
  drag: float = 0.
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  _forcing_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky, self.kz = self.grid.rfft_mesh()
    self.laplace = (2j * jnp.pi)**2 * (self.kx**2 + self.ky**2 + self.kz**2)
    self.filter_ = spectral_utils3d.brick_wall_filter_3d(self.grid)
    self.linear_term = self.viscosity * self.laplace - self.drag
    self.linear_term00 = self.viscosity * (2j * jnp.pi * self.ky[0, :, 0])**2 - self.drag

    # setup the forcing function with the caller-specified grid.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)

  def explicit_terms(self, xvec_hat, screenout = False):
    laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat = xvec_hat

    # recompute the velocity components from omy and Lapl(vy)
    velocity_solve = spectral_utils3d.KMM_velvort_to_velocity(self.grid)
    vxhat, vyhat, vzhat, omx_hat, omz_hat = velocity_solve(laplvy_hat, omy_hat, \
                                                           vx00_hat, vy00_hat, vz00_hat)


    # transform velocity and vorticity components to real space
    vx, vy, vz, omx, omy, omz = jnp.fft.irfftn(vxhat),   \
                                jnp.fft.irfftn(vyhat),   \
                                jnp.fft.irfftn(vzhat),   \
                                jnp.fft.irfftn(omx_hat), \
                                jnp.fft.irfftn(omy_hat), \
                                jnp.fft.irfftn(omz_hat)

    # DEBUGGING & screenoutput: check box-averaged kinetic energy etc:
    if screenout:
      div3d_hat = 2j * jnp.pi * (self.kx * vxhat + self.ky * vyhat + self.kz * vzhat)
      div3d = jnp.fft.irfftn(div3d_hat)
      eneruvw = (vx**2 + vy**2 + vz**2)/2.
      enstromxyz = (omx**2 + omy**2 + omz**2)/2.
      jax.debug.print("div(v)= {var}", var=div3d.max(axis=None))
      jax.debug.print("max(u)= {var}", var=vx.max(axis=None))
      jax.debug.print("max(v)= {var}", var=vy.max(axis=None))
      jax.debug.print("max(w)= {var}", var=vz.max(axis=None))
      jax.debug.print("eneruvw= {var}", var=eneruvw.mean(axis=None))
      jax.debug.print("enstromxyz= {var}", var=enstromxyz.mean(axis=None))
      #return

    # now compute the advection terms H1/2/3 in real space in rotational form
    # -> Hi = -(u.grad)u = omega x u + 1/2 * grad.|u|^2
    #    (where the second term can be absorbed into the pressure gradient)
    advectionx = -(omy * vz - omz * vy)
    advectiony = -(omz * vx - omx * vz)
    advectionz = -(omx * vy - omy * vx)

    # transform back to Fourier space
    advectionx_hat, \
    advectiony_hat, \
    advectionz_hat = jnp.fft.rfftn(advectionx), \
                     jnp.fft.rfftn(advectiony), \
                     jnp.fft.rfftn(advectionz)

    # smooth the advection terms according to the 2/3 rule
    if self.smooth is not None:
      advectionx_hat *= self.filter_
      advectiony_hat *= self.filter_
      advectionz_hat *= self.filter_

    # compute the non-linear terms in Fourier space
    # (i) (laplvy-equ): -d/dy [d(H1)/dx + d(H3)/dz] + (d/dxx + d/dzz) H2
    nl_laplvy = -2j * jnp.pi * self.ky * \
              (  2j * jnp.pi * self.kx * advectionx_hat + \
                 2j * jnp.pi * self.kz * advectionz_hat ) \
              + (2j * jnp.pi)**2 * (self.kx**2 + self.kz**2) * advectiony_hat

    # (ii) (omy-equ):  [d(H1)/dz - d(H3)/dx]
    nl_omy = 2j * jnp.pi * self.kz * advectionx_hat \
           - 2j * jnp.pi * self.kx * advectionz_hat

    laplvy_terms = nl_laplvy
    omy_terms    = nl_omy

    # next, we compute the explicit terms of the original momentum equation for the kx=kz=0 modes
    vx00_terms = advectionx_hat[0, :, 0]
    vy00_terms = advectiony_hat[0, :, 0] * 0. # H_2 cancels vs. d(p_00)/dy
    vz00_terms = advectionz_hat[0, :, 0]

    # add forcing term to the RHS
    if self.forcing_fn is not None:
      fx, fy, fz = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                               _get_grid_variable(vy, self.grid),
                                               _get_grid_variable(vz, self.grid)))

      if screenout:
        einp = (fx.data*vx + fy.data*vy + fz.data*vz)
        jax.debug.print("Einp= {var}", var=einp.mean(axis=None))
        return

      fx_hat, fy_hat, fz_hat = jnp.fft.rfftn(fx.data), \
                               jnp.fft.rfftn(fy.data), \
                               jnp.fft.rfftn(fz.data)

      # Remark: fi with i \in {x,y,z} can be treated in analogy to the non-linear
      #         terms Hi above, that is:
      # (i) (laplvy-equ): -d/dy [d(f1)/dx + d(f3)/dz] + (d/dxx + d/dzz) f2
      laplvy_terms += -2j * jnp.pi * self.ky * \
                    (  2j * jnp.pi * self.kx * fx_hat + \
                       2j * jnp.pi * self.kz * fz_hat ) \
                    + (2j * jnp.pi)**2 * (self.kx**2 + self.kz**2) * fy_hat

      # (ii) (omy-equ):  [d(f1)/dz - d(f3)/dx]
      omy_terms += 2j * jnp.pi * self.kz * fx_hat \
                 - 2j * jnp.pi * self.kx * fz_hat

      # (iii) (mom-equ): fi for the kx=kz=0 - modes
      vx00_terms += fx_hat[0, :, 0]
      vy00_terms += fy_hat[0, :, 0]
      vz00_terms += fz_hat[0, :, 0]

    # TODO: still experimental: We set the RHS of the kx=ky=kz=0 mode to 0 to
    # ensure that d(u/v/w_000)/t = 0 and hence u/v/w_000 = cst. in time
    vx00_terms = vx00_terms.at[0].set(0j)
    vy00_terms = vy00_terms.at[0].set(0j)
    vz00_terms = vz00_terms.at[0].set(0j)

    return (laplvy_terms, omy_terms, vx00_terms, vy00_terms, vz00_terms)

  def implicit_terms(self, xvec_hat):
    laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat = xvec_hat
    return (self.linear_term * laplvy_hat,
            self.linear_term * omy_hat,
            self.linear_term00 * vx00_hat,
            self.linear_term00 * vy00_hat,
            self.linear_term00 * vz00_hat)

  def implicit_solve(self, xvec_hat, time_step):
    laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat = xvec_hat
    return (1 / (1 - time_step * self.linear_term) * laplvy_hat,
            1 / (1 - time_step * self.linear_term) * omy_hat,
            1 / (1 - time_step * self.linear_term00) * vx00_hat,
            1 / (1 - time_step * self.linear_term00) * vy00_hat,
            1 / (1 - time_step * self.linear_term00) * vz00_hat)




def ForcedNavierStokes3D(viscosity, kf, grid, smooth):


  #offsets = ((0, 0), (0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: forcings.kolmogorov_forcing(
      grid, scale=viscosity, kf=kf, forcedir=0, forcing2d=False)
  return NavierStokes3D(
      viscosity,
      grid,
      drag=0.,
      smooth=smooth,
      forcing_fn=forcing_fn)


