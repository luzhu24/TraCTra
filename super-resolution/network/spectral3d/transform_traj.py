import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import utils


def rft_trajectory_to_physical(rft_traj: tuple, grid: cfd.grids.Grid) -> tuple:
  velocity_solve = utils.KMM_velvort_to_velocity(grid) 
  laplvy_hat, omy_hat, vx00_hat, vy00_hat, vz00_hat = rft_traj
  vxhat = jnp.empty_like(laplvy_hat)
  vyhat = jnp.empty_like(laplvy_hat)
  vzhat = jnp.empty_like(laplvy_hat)
  #
  nSnapsTraj = laplvy_hat.shape[0]
  nSpaceDim  = len(laplvy_hat.shape[1:])

  for iSnap in range(nSnapsTraj):
    laplvy_hat_snap = laplvy_hat[iSnap, :, :, :]
    omy_hat_snap    = omy_hat[iSnap, :, :, :]
    vx00_hat_snap   = vx00_hat[iSnap, :]
    vy00_hat_snap   = vy00_hat[iSnap, :]
    vz00_hat_snap   = vz00_hat[iSnap, :]
    #
    vxhat_snap, vyhat_snap, \
    vzhat_snap, omx_hat_snap, \
    omz_hat_snap = velocity_solve(laplvy_hat_snap, omy_hat_snap, \
                                  vx00_hat_snap, vy00_hat_snap, vz00_hat_snap)                

    vxhat = vxhat.at[iSnap, :, :, :].set(vxhat_snap) 
    vyhat = vyhat.at[iSnap, :, :, :].set(vyhat_snap) 
    vzhat = vzhat.at[iSnap, :, :, :].set(vzhat_snap) 

  vx, vy, vz = jnp.fft.irfftn(vxhat, axes=tuple(range(1, nSpaceDim + 1))), \
               jnp.fft.irfftn(vyhat, axes=tuple(range(1, nSpaceDim + 1))), \
               jnp.fft.irfftn(vzhat, axes=tuple(range(1, nSpaceDim + 1)))
  
  return vx, vy, vz
