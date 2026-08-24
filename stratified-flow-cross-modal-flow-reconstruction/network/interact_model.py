import jax.numpy as jnp




def real_to_real_traj_fn_stf3d(vel_phys, vel2vort_fn, vort2vel_fn_t, traj_fn):

  vort_rft = vel2vort_fn(vel_phys)

  _, traj_rft = traj_fn(vort_rft) 

  traj_phys_t = vort2vel_fn_t(traj_rft) 

  return traj_phys_t



def vel2vort(vel_phys, KMM_vel2vort_fn):

  r_rft = jnp.fft.rfftn(vel_phys[...,-1],axes=(-3,-2,-1))
  vort_rft = KMM_vel2vort_fn(vel_phys[...,0],vel_phys[...,1],vel_phys[...,2],)
  vort_rft = vort_rft + (r_rft,)

  return vort_rft


def vort2vel(vor_rft, KMM_vort2vel_fn):

  r_rft = vor_rft[-1] 
  vel_rft = KMM_vort2vel_fn(vor_rft[0],vor_rft[1],vor_rft[2],vor_rft[3],vor_rft[4],)
  vel_rft = vel_rft[:3] + (r_rft,) 

  vel_rft = jnp.stack(vel_rft, axis=0) 
  vel_phy = jnp.fft.irfftn(vel_rft, axes=(-3,-2,-1))
  vel_phy = jnp.moveaxis(vel_phy, 0, -1)

  return vel_phy




def average_pool_trajectory(omega_traj, pool_width, pool_height):
  trajectory_length, Nx, Ny, Nchannels = omega_traj.shape
  assert Nx % pool_width == 0
  assert Ny % pool_height == 0

  omega_reshaped = omega_traj.reshape(
    (trajectory_length, Nx // pool_width, pool_width, Ny // pool_height, pool_height, Nchannels)
  )
  omega_pooled_traj = omega_reshaped.mean(axis=(2, 4))
  return omega_pooled_traj

def coarse_pool_trajectory(omega_traj, pool_width, pool_height):
  _, Nx, Ny, _ = omega_traj.shape
  assert Nx % pool_width == 0
  assert Ny % pool_height == 0
  coarse_x = pool_width
  coarse_y = pool_height

  omega_pooled_traj = omega_traj[:, ::coarse_x, ::coarse_y, :]
  return omega_pooled_traj
