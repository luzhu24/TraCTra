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





def traj_VC_newt3d_sresol(
    vor_true_traj: jnp.ndarray,
    pred_traj: jnp.ndarray,
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    vmax: float = 2,
    truth_weight_kind: str = "uniform",
    traj_weight_kind: str = "linear_late",
    weight_strength: float = 1.0,):
    """
    Time-weighted loss.
    Assumes arrays have shape (B, T, ..., C_like)
    """
    eps = 1e-8

    # rollout from predicted initial state
    v0 = vmax*jnp.tanh(pred_traj[:,0,...]/vmax)
    vor0_pred_traj = trajectory_rollout_fn(v0)


    T = vor_true_traj.shape[1]
    Tm1 = T #- 1

    w_true = make_time_weights(T, kind=truth_weight_kind, strength=weight_strength)
    w_traj = make_time_weights(Tm1, kind=traj_weight_kind, strength=weight_strength)


    # time-dependent terms
    se_V_true = (vor_true_traj - pooling_fn(vor0_pred_traj)) ** 2
    se_V_traj = (pred_traj - jax.lax.stop_gradient(vor0_pred_traj)) ** 2

    mse_true = weighted_time_mse(se_V_true, w_true)
    mse_V_traj = weighted_time_mse(se_V_traj, w_traj)


    # normalization factors
    E_v_true = jnp.mean(vor_true_traj ** 2)

    V_traj_ref = jax.lax.stop_gradient(vor0_pred_traj)

    E_V_ref = jnp.mean(V_traj_ref ** 2)

    L_v_true_rel = mse_true / (E_v_true + eps)
    L_v_traj_rel = mse_V_traj / (E_V_ref + eps)

    return alpha * L_v_true_rel + beta * L_v_traj_rel





def fulldata_newt3d_sresol(
    vor_true_traj: jnp.ndarray,
    pred_traj: jnp.ndarray,
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    vmax: float = 2,
    truth_weight_kind: str = "uniform",
    traj_weight_kind: str = "linear_late",
    weight_strength: float = 1.0,):
    """
    Time-weighted loss.
    Assumes arrays have shape (B, T, ..., C_like)
    """
    eps = 1e-8

    # rollout from predicted initial state
    #v0 = vmax*jnp.tanh(pred_traj[:,0,...]/vmax)
    #vor0_pred_traj = trajectory_rollout_fn(v0)


    T = vor_true_traj.shape[1]
    w_true = make_time_weights(T, kind=truth_weight_kind, strength=weight_strength)

    # time-dependent terms
    se_V_true = (vor_true_traj - pred_traj) ** 2
    mse_true = weighted_time_mse(se_V_true, w_true)

    # normalization factors
    E_v_true = jnp.mean(vor_true_traj ** 2)

    L_v_true_rel = mse_true / (E_v_true + eps)

    return alpha * L_v_true_rel


