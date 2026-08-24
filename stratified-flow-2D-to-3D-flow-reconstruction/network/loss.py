import jax
import jax.numpy as jnp
from typing import Callable



def weighted_time_mse(se: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    se_t = jnp.mean(se, axis=tuple(range(2, se.ndim)))   # (B, T)
    return jnp.sum(se_t * w[None, :]) / (se_t.shape[0] * (jnp.sum(w) + eps))


def make_time_weights(T: int, kind: str = "linear_late", strength: float = 1.0):
    t = jnp.linspace(0.0, 1.0, T)
    if kind == "uniform":
        w = jnp.ones((T,))
    elif kind == "linear_late":
        w = 1.0 + strength * t
    elif kind == "exp_late":
        w = jnp.exp(strength * t)
    elif kind == "linear_early":
        w = 1.0 + strength * (1.0 - t)
    else:
        raise ValueError(f"Unknown weight kind: {kind}")
    return w / jnp.mean(w)





def traj_VC_stf3d_slice2vr(
    r_true_traj: jnp.ndarray, 
    pred_traj: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    slicing_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 1,
    beta: float = 1,
    vmax: float = 0.5,
    rmax: float = 1,
    truth_weight_kind: str = "uniform",
    traj_weight_kind: str = "linear_late",
    weight_strength: float = 1.0,
    ):
  """  Loads in full trajectories (with noise) for comparison 
       vor_true_traj.shape = (batch_size, time, Nx, Nz, 1)
       pred_traj.shape = (batch_size, time, Nx, Ny, Nz, 4) """

  vr0_pred = pred_traj[:,0,...]

  v0 = vmax*jnp.tanh(vr0_pred[..., :3]/vmax)
  r0 = rmax*jnp.tanh(vr0_pred[..., 3:]/rmax)
  vr0_safe = jnp.concatenate([v0, r0], axis=-1)

  T = r_true_traj.shape[1]
  Tm1 = T #- 1

  w_true = make_time_weights(T, kind=truth_weight_kind, strength=weight_strength)
  w_traj = make_time_weights(Tm1, kind=traj_weight_kind, strength=weight_strength)


  vr0_pred_traj = trajectory_rollout_fn(vr0_safe)
  r0_pred_traj = vr0_pred_traj[...,3:]

  r0_pred_traj_slice = slicing_fn(r0_pred_traj)

  eps = 1e-6
  se_r_true = (r_true_traj - r0_pred_traj_slice) ** 2 
  se_vr_traj = (pred_traj - jax.lax.stop_gradient(vr0_pred_traj)) ** 2 


  mse_r = weighted_time_mse(se_r_true, w_true)
  mse_vr = weighted_time_mse(se_vr_traj, w_traj)

  E_r_true = jnp.mean(r_true_traj**2)

  vr_ref = jax.lax.stop_gradient(vr0_pred_traj) 
  E_vr_ref = jnp.mean(vr_ref**2)  

  L_r_rel = mse_r / (E_r_true + eps)
  L_v_rel   = mse_vr   / (E_vr_ref   + eps)

  return alpha * L_r_rel + beta * L_v_rel


