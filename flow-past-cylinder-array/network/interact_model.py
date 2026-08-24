import jax.numpy as jnp

def real_to_real_traj_fn(vort_phys, traj_fn):
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(1,2))[...,0]
  _, traj_rft = traj_fn(vort_rft) 
  traj_phys = jnp.fft.irfftn(traj_rft, axes=(2,3))[...,jnp.newaxis]
  return traj_phys


def real_to_real_traj_fn_fpc(vort_phys, traj_fn):

  vort_phys,U0 = vort_phys
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(1,2))[...,0]
  _, traj_rft = traj_fn([vort_rft,U0]) # traj_rft[t,traj,x,y] --LZ

  traj_rft,UU = traj_rft
  traj_phys = jnp.fft.irfftn(traj_rft, axes=(2,3))[...,jnp.newaxis]
  return [traj_phys,UU]



def compute_vel_traj(
    vort_traj: jnp.ndarray,
    dx: float, 
    dy: float
) -> jnp.ndarray:
  _, Nx, Ny, _ = vort_traj.shape
  vort_traj_rft = jnp.fft.rfftn(vort_traj, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis]
  ky_mesh = (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis]
  
  psik = vort_traj_rft / (1e-7 + kx_mesh ** 2 + ky_mesh ** 2)
  
  uk =  1j * ky_mesh * psik
  vk = -1j * kx_mesh * psik

  u_traj = jnp.fft.irfftn(uk, axes=(1,2))
  v_traj = jnp.fft.irfftn(vk, axes=(1,2))
  vel_traj = jnp.concatenate([u_traj, v_traj], axis=-1)
  return vel_traj

def compute_vort_traj(
    vel_traj: jnp.ndarray,
    dx: float,
    dy: float
) -> jnp.ndarray:
  _, Nx, Ny, _ = vel_traj.shape
  vel_traj_rft = jnp.fft.rfftn(vel_traj, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = (kx_mesh.T)[jnp.newaxis, ...]
  ky_mesh = (ky_mesh.T)[jnp.newaxis, ...]

  vort_traj_rft = 1j * kx_mesh * vel_traj_rft[..., 1] - 1j * ky_mesh * vel_traj_rft[..., 0]
  vort_traj = jnp.fft.irfftn(vort_traj_rft, axes=(1,2))
  return vort_traj[..., jnp.newaxis]


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
