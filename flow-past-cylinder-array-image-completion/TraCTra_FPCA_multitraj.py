import os
import glob
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import keras
import jax_cfd.base as cfd
from functools import partial

from network import models
from network import time_stepping as ts
from network import loss as lf
from network import interact_model as im

from signed_distance import signed_distance as sd


# -------------------------
# I/O
# -------------------------
def loadnpz(field_names, filename="./data.npz", mmap=False, dtype=np.float32):
    with np.load(filename, mmap_mode="r" if mmap else None) as dnsdata:
        arrays = []
        for name in field_names:
            arr = dnsdata[name]
            arr = np.asarray(arr, dtype=dtype)
            arrays.append(arr)

        fields = arrays[0][..., None] if len(arrays) == 1 else np.stack(arrays, axis=-1)
        U = np.asarray(dnsdata["U"], dtype=dtype)
        param = np.asarray(dnsdata["param"])
        geom = np.asarray(dnsdata["geom"])
    return (fields, U), param, geom


def loadnpz_multitraj(field_names, data_loc, file_front, file_end=".npz",
                      n_files=None, mmap=False, dtype=np.float32):
    """Load multiple FPC trajectories with possibly different time lengths.
    """
    pattern = os.path.join(data_loc, f"{file_front}_*{file_end}")
    filenames = sorted(glob.glob(pattern))
    print (filenames)

    if n_files is not None:
        filenames = filenames[:n_files]

    traj_lengths = []
    field_shape = None
    param_ref = None
    geom_ref = None

    for i, filename in enumerate(filenames):
        (fields_i, U_i), param_i, geom_i = loadnpz(field_names, filename, mmap=mmap, dtype=dtype)
        if i == 0:
            field_shape = fields_i.shape[1:]
            param_ref = np.array(param_i, copy=True)
            geom_ref = np.array(geom_i, copy=True)
        traj_lengths.append(fields_i.shape[0])

    total_snapshots = int(np.sum(traj_lengths))
    fields = np.empty((total_snapshots,) + field_shape, dtype=dtype)
    U = np.empty((total_snapshots,), dtype=dtype)

    array_offset = 0
    for i, (filename, nt_i) in enumerate(zip(filenames, traj_lengths)):
        (fields_i, U_i), _, _ = loadnpz(field_names, filename, mmap=mmap, dtype=dtype)
        fields[array_offset:array_offset + nt_i] = fields_i
        U[array_offset:array_offset + nt_i] = U_i
        array_offset += nt_i
        print(
            f"Loaded {i + 1:4d}/{len(filenames)}: {filename}, "
            f"Nt={nt_i}, accumulated={array_offset}")

    geom = np.array(geom_ref, copy=True)
    geom[-2] = total_snapshots

    print(f"Trajectory lengths: {traj_lengths}")
    print(
        f"Combined {len(filenames)} trajectories into initial conditions: "
        f"fields={fields.shape}, U={U.shape}")
    return (fields, U), param_ref, geom



def mask_function(m, mask=None):
    return m * mask[..., None]



# -------------------------
# Generator
# -------------------------
def keras_gen_offset(arr, idx, batch_size, *,
                     traj_rollout, pool_fn, obs_mask, geom_chan, offset=0,
                     shuffle=True, rng=None):
    """Roll out the full FPC state, then build forward-offset observations.
    """
    if rng is None:
        rng = np.random.default_rng()

    ff, UU = arr

    # Prepare static channels once.
    obs_mask = np.asarray(obs_mask, dtype=np.float32)[None, None, ..., None]
    geom_chan = np.asarray(1 - geom_chan, dtype=np.float32)[None, None, ..., None]

    while True:
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, order.size, batch_size):
            b = order[s:s + batch_size]
            batch = np.asarray(ff[b], dtype=np.float32)
            batch_U = np.asarray(UU[b], dtype=np.float32)

            rollout_out = traj_rollout((batch, batch_U))

            batch_traj = np.asarray(rollout_out[0], dtype=np.float32)
            batch_U_traj = np.asarray(rollout_out[1], dtype=np.float32)

            Nt_ext = batch_traj.shape[1]
            Nt = Nt_ext - offset

            obs_traj = np.asarray(pool_fn(batch_traj), dtype=np.float32)

            # Forward-only.
            if offset == 0:
                x = obs_traj[:, :Nt, ...]
            else:
                parts = [obs_traj[:, i:i + Nt, ...]
                         for i in range(offset + 1)]
                x = np.concatenate(parts, axis=-1)

            y = obs_traj[:, :Nt, ...]
            y_U = batch_U_traj[:, :Nt, None, None, None]

            obs = np.broadcast_to(obs_mask, x.shape[:-1] + (1,))
            geom = np.broadcast_to(geom_chan, x.shape[:-1] + (1,))
            y_U = np.broadcast_to(y_U, y.shape[:-1] + (1,))

            x = np.concatenate([x, obs, geom], axis=-1).astype(np.float32)
            y = np.concatenate([y, y_U], axis=-1).astype(np.float32)

            yield x, y


# -------------------------
# Main
# -------------------------
data_loc = "../data/training/"
weight_loc = "./"
file_front = "data_fpc"
file_end = ".npz"
n_files = None

loadweights = False
weight_name = "CNN_best_imcomp_V_wh11.multitraj.weights.h5"


snapshots, param, geom = loadnpz_multitraj(["vor",], data_loc, file_front, file_end, n_files=n_files, mmap=False)
Re,fforce = param
Nx, Ny, Lx, Ly, Nt, dt_dns = geom
Nx, Ny = int(Nx), int(Ny)
dt_dns = float(dt_dns)
Lx = float(Lx); Ly = float(Ly)
print("geom:", geom, "param:", param, "snapshots:", snapshots[0].shape)
smooth = False


# mask function
ndx = 2
delta = Lx/Nx*ndx
delta_hester = 2.64822828 # 
eta = Re*(delta/delta_hester)**2
obj1 = sd.rounded_rectangle_segments(cx=-1, cy=0.0, w=1, h=1, r=0.1)
obj_obs = sd.rectangle_segments(cx=1, cy=0.0, w=1, h=1)


# -------------------------
# Hyperparameters
# -------------------------
raw_channels = snapshots[0].shape[-1] 
output_channels=1
T_unroll = 20.
T_interval = 2
stride = 1
dt_stable = 0.05 
alphal,betal=1,1
t_offset = 4
input_channels = raw_channels * (t_offset + 1) + 2

# training hyp
batch_size = 8
lr_traj = 1e-4
nval = 5
n_traj_steps = 5000
n_layer = 1
n_level = 5
kernel = (3,3)
n_filters=16
filter_factor=1.5


grid = cfd.grids.Grid((Nx, Ny), domain=((-Lx/2., Lx/2.), (-Ly / 2.0, Ly / 2.0)))

offsets = (0, 0)
x, y = grid.axes(offsets)

# mask function of objects
sdf_obj = sd.SignedDistanceFromSegments2D(x=x, y=y, Lx=Lx, Ly=Ly)
sdf_obj.set_objects([obj1, ]).build(union_mode="outer", tol=(x[1]-x[0]), do_reinit=False, periodic_x=True, periodic_y=True)
geom_mask=jnp.asarray(sdf_obj.phi)
geom_chan = 0.5*(1+jnp.tanh(2./delta*sdf_obj.phi))

# mask function of observation
sdf_obs = sd.SignedDistanceFromSegments2D(x=x, y=y, Lx=Lx, Ly=Ly)
sdf_obs.set_objects([obj_obs, ]).build(union_mode="outer", tol=(x[1]-x[0]), do_reinit=False, periodic_x=True, periodic_y=True)
obs_mask = np.asarray(sdf_obs.phi >= 0, dtype=np.float32)


mask_fn = jax.jit(partial(mask_function,mask=obs_mask))
mask_fn_B_t = jax.vmap(jax.vmap(mask_fn))


snapshots = (snapshots[0][::stride], snapshots[1][::stride])
print("Initial-condition snapshots:", snapshots[0].shape, snapshots[1].shape)

# -------------------------
# Trajectory rollout function (physics)
# -------------------------
t_substep = int(round(T_interval/dt_stable))
n_snapshots = int(round(T_unroll/T_interval))+1
n_snapshots_offset = n_snapshots + t_offset

print(
    "dt_stable:", dt_stable,
    "t_substep:", t_substep,
    "n_snapshots:", n_snapshots,
    "n_snapshots_offset:", n_snapshots_offset,
)

trajectory_fn = ts.generate_trajectory_fn_FPC(Re, fforce, geom_mask, eta, delta, dt_stable, grid, t_substep=t_substep, n_substep=n_snapshots, smooth=smooth)
trajectory_fn_offset = ts.generate_trajectory_fn_FPC(Re, fforce, geom_mask, eta, delta, dt_stable, grid, t_substep=t_substep, n_substep=n_snapshots_offset, smooth=smooth)

real_traj_fn = partial(im.real_to_real_traj_fn_fpc, traj_fn=jax.vmap(trajectory_fn))
real_traj_offset_fn = partial(im.real_to_real_traj_fn_fpc, traj_fn=jax.vmap(trajectory_fn_offset))

# -------------------------
# Model
# -------------------------
newt_model = models.newt_VC_traj_fpc(Nx, Ny, n_snapshots,
    N_filters=n_filters, N_layer=n_layer, N_levels=n_level, kernel=kernel, 
    input_channels=input_channels, output_channels=output_channels,
    filter_factor=filter_factor)


loss_fn = jax.jit(partial(lf.mse_and_traj_vor_weighted,
                        trajectory_rollout_fn=real_traj_fn,
                        pooling_fn=mask_fn_B_t,
                        alpha=alphal, beta=betal,
                        vmax=5,
                        truth_weight_kind = "uniform",
                        traj_weight_kind = "exp_late",
                        weight_strength = 1.0,))


newt_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_traj), loss=loss_fn,)
if loadweights:
    print("loading weights...")
    newt_model.load_weights(weight_loc + weight_name)


# -------------------------
# Train/val split (indices over initial-condition snapshots)
# -------------------------
rng = np.random.default_rng(42)
Nwin = snapshots[0].shape[0]
perm = rng.permutation(Nwin)

train_idx = perm[:-nval]
val_idx = perm[-nval:]

steps_per_epoch = int(np.ceil(train_idx.size / batch_size))
val_steps = int(np.ceil(val_idx.size / batch_size))




train_gen = keras_gen_offset(snapshots, train_idx, batch_size, traj_rollout=real_traj_offset_fn, pool_fn=mask_fn_B_t,obs_mask=obs_mask, geom_chan=geom_chan, offset=t_offset, shuffle=True, rng=np.random.default_rng(1),)

val_gen = keras_gen_offset(snapshots, val_idx, batch_size, traj_rollout=real_traj_offset_fn, pool_fn=mask_fn_B_t,  obs_mask=obs_mask, geom_chan=geom_chan, offset=t_offset, shuffle=False, rng=np.random.default_rng(2),)


# -------------------------
# Training loop + checkpointing
# -------------------------
min_val_loss = np.inf
hist_loss, hist_val_loss = [], []
for n in range(n_traj_steps):
    print("Traj step:", n,flush=True)
    history = newt_model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=1, verbose=2,)

    current_loss = float(history.history["loss"][-1])
    current_val_loss = float(history.history["val_loss"][-1])
    hist_loss.append(current_loss)
    hist_val_loss.append(current_val_loss)

    if (not np.isfinite(current_loss)) or (not np.isfinite(current_val_loss)):
        print("Non-finite loss detected. Stopping training.")
        break

    if current_val_loss < min_val_loss:
        min_val_loss = current_val_loss
        newt_model.save_weights(weight_loc + weight_name)


np.savez(data_loc + "history_multitraj.npz", loss=np.array(hist_loss), val_loss=np.array(hist_val_loss))

