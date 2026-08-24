import jax
import jax.numpy as jnp
from typing import Callable






def weighted_time_mse(se: jnp.ndarray, w: jnp.ndarray, m: jnp.ndarray = None, eps: float = 1e-8) -> jnp.ndarray:
    if m is None:
        se_t = jnp.mean(se, axis=tuple(range(2, se.ndim)))   # (B, T)
    else:
        num = jnp.sum(se * m, axis=tuple(range(2, se.ndim)))
        den = jnp.sum(m, axis=tuple(range(2, m.ndim)))
        se_t = num / (den + eps)   # (B, T)

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



def mse_and_traj_vor_weighted(
    true_traj: jnp.ndarray,
    pred_traj: jnp.ndarray,
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    vmax: float = 1.0,
    truth_weight_kind: str = "uniform",
    traj_weight_kind: str = "linear_late",
    weight_strength: float = 1.0,):
    """
    Time-weighted loss.
    """
    eps = 1e-8


    vor_true_traj = true_traj[...,:-1]
    U_true_traj = true_traj[...,0,0,-1]


    vor0_pred = pred_traj[:,0,...]

    vor0_safe = vmax*jnp.tanh(vor0_pred/vmax)

    vor0_pred_traj,U0_traj = trajectory_rollout_fn([vor0_safe,U_true_traj[:,0]])

    T = vor_true_traj.shape[1]
    Tm1 = T #- 1

    w_true = make_time_weights(T, kind=truth_weight_kind, strength=weight_strength)
    w_traj = make_time_weights(Tm1, kind=traj_weight_kind, strength=weight_strength)

    mask = pooling_fn(jnp.ones_like(vor_true_traj))

    # time-dependent terms
    se_true = pooling_fn((vor_true_traj - vor0_pred_traj)**2)
    se_traj = (vor0_pred_traj - pred_traj) ** 2

    mse_true = weighted_time_mse(se_true, w_true, m=mask)
    mse_traj = weighted_time_mse(se_traj, w_traj)

    E_v_true = jnp.sum(vor_true_traj ** 2)/(jnp.sum(mask)+eps)

    V_traj_ref = jax.lax.stop_gradient(vor0_pred_traj)

    E_V_ref = jnp.mean(V_traj_ref ** 2)

    L_true_rel = mse_true / (E_v_true + eps)
    L_traj_rel = mse_traj / (E_V_ref + eps)

    return alpha * L_true_rel + beta * L_traj_rel




