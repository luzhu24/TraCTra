import jax
import jax.numpy as jnp
from typing import Callable





def traj_VC_stf3d_r2v(
    r_true_traj: jnp.ndarray, 
    pred_traj: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 1,
    beta: float = 1,
    vmax: float = 0.5,
    ):
  """  Loads in full trajectories (with noise) for comparison 
       vor_true_traj.shape = (batch_size, time, Nx, Ny, Nz, 1)
       pred_traj.shape = (batch_size, time, Nx, Ny, Nz, 4) """

  vr0_pred = pred_traj[:,0,...]
  v_pred = pred_traj[...,:3]

  v0 = vmax*jnp.tanh(vr0_pred[..., :3]/vmax)
  vr0_safe = jnp.concatenate([v0, vr0_pred[..., 3:]], axis=-1)

  vr0_pred_traj = trajectory_rollout_fn(vr0_safe)
  v0_pred_traj = vr0_pred_traj[...,:3]
  r0_pred_traj = vr0_pred_traj[...,3:]

  eps = 1e-6
  se_r_traj = (r_true_traj[:,1:,...] - r0_pred_traj[:,1:,...]) ** 2 
  se_v_traj = (v_pred[:,1:,...] - v0_pred_traj[:,1:,...]) ** 2 

  mse_r = jnp.mean(se_r_traj)
  mse_v = jnp.mean(se_v_traj)

  E_r_true = jnp.mean(r_true_traj[:, 1:, ...]**2)

  v_ref = jax.lax.stop_gradient(v0_pred_traj[:, 1:, ...])  
  E_v_ref = jnp.mean(v_ref**2)  

  L_r_rel = mse_r / (E_r_true + eps)
  L_v_rel   = mse_v   / (E_v_ref   + eps)

  return alpha * L_r_rel + beta * L_v_rel


