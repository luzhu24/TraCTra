""" Train models on noisy trajectories (coarse approach and velocity ONLY)
    different dataset required since training loads full trajectories
    rather than snapshots. Single run with fixed dt etc. """
import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


import yaml

import keras
import jax_cfd.base as cfd

from functools import partial

from network import models
from network import time_stepping as ts
from network import loss as lf
from network import interact_model as im
from network import sym_augment as sa

from network.spectral3d import utils


# Use parameters from config file
def loadnpz(fieldname,filename='./data.npz'):
    fields=[]
    dnsdata = np.load(f'{filename}')
    for fn in fieldname:
        fields.append(np.asarray(dnsdata[fn],dtype=np.float32))#[field,nt,nx,ny,nz]
    param = dnsdata['param']
    geom = dnsdata['geom']
    fields = np.array(fields).transpose((1,2,3,4,0))#[nt,nx,ny,nz,field]
    return fields,param,geom



def make_subtrajectories(A, dT, dt_dns, stride=1, T_interval=1):
    """
    series: (T, Ns, C) sensor time series
    returns:
      windows: view of shape (Nwin, Nt_win, Ns, C) (typically zero-copy)
      Nt_win: int number of time steps in each window after T_interval
    """
    A = np.asarray(A)
    T = A.shape[0]
    L = int(np.floor(dT / dt_dns))
    if T < L:
        raise ValueError(f"Not enough time frames: T={T} < L={L}")

    W = sliding_window_view(A, window_shape=L, axis=0)  
    W = np.moveaxis(W, -1, 1)                       
    W = W[::stride, ::T_interval]                         
    return W, W.shape[1]




# -------------------------
# Generator (no big trajectory arrays)
# -------------------------
def keras_gen_sr_pretrain(arr, idx, batch_size, *,
                          pooling_fn_batched,
                          shuffle=True,
                          rng=None):
    """
    arr: (Nwin, Nt, Nx, Ny, C) high-res windows
    Return:
      x: coarse input (B, Nt, Nx_c, Ny_c, n_input_ch)
      y: high-res target (B, Nt, Nx, Ny, n_input_ch), built from bilinear upsampling
         of the coarse input (so no DNS needed conceptually).
    """
    if rng is None:
        rng = np.random.default_rng()

    while True:
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, order.size, batch_size):
            b = order[s:s + batch_size]
            batch = arr[b].astype(np.float32, copy=False)  

            x_fine = batch[...]

            x_coarse = pooling_fn_batched(jnp.asarray(x_fine))  

            yield x_coarse, x_coarse





data_loc = '/mnt/ceph_rbd/flow3d/DNS/'
weight_loc = '/mnt/ceph_rbd/flow3d/newt3d_re10000/'
file_front = 'data_newt_re10000_long'
file_end = '.npz'
n_files = 200

loadweights = False
weight_name = 'newt3d_best_traj_V1C1_lev4.weights.test.h5'
snapshots,param,geom = loadnpz(['vx','vy','vz'],data_loc+file_front+file_end)
re,kf = param
Nx,Ny,Nz,Lx,Ly,Lz,Nt,dt_dns = geom
Nx,Ny,Nz,Nt = int(Nx),int(Ny),int(Nz),int(Nt)
smooth = True
print (geom,param,snapshots.shape)

noise_level = 0.1

# -------------------------
# Hyperparameters
# -------------------------
dt_stable = 0.2
noise_level = 0.0
T_unroll = 10.0
T_interval = 1
stride = 1
batch_size = 1
lr_traj = 1e-4
nval = 5
n_traj_steps = 1000
alphal,betal = 1,1

N_grow = 4
filter_size = 2**N_grow
N_filters=16 
N_layer=2
N_deep=4
kernel=(3,3,3)
input_channels=3
output_channels=3

Nx_coarse, Ny_coarse, Nz_coarse = Nx // filter_size, Ny // filter_size, Nz // filter_size


grid = cfd.grids.Grid((Nx, Ny, Nz), domain=((0, Lx), (-Ly/2., Ly/2.), (-Lz/2., Lz/2.)))

max_vel_est = 1.

snapshots_trajs,n_snapshots = make_subtrajectories(snapshots, T_unroll, dt_dns, stride=stride, T_interval=T_interval)
print (snapshots_trajs.shape,n_snapshots)

min_val_loss = np.inf

t_substep = (dt_dns*T_interval)/dt_stable
N_substep = t_substep*n_snapshots
print("dt_stable:", dt_stable, "T substep:", t_substep, "N_substep:", N_substep)
trajectory_fn = ts.generate_trajectory_fn_newt3d(re,kf, dt_stable, grid, t_substep=t_substep, n_substep=n_snapshots, smooth=smooth)


KMMvel2vort = utils.KMM_velocity_to_velvort(grid)
vel2vort_fn = jax.jit(partial(im.vel2vort_newt, KMM_vel2vort_fn=KMMvel2vort))


KMMvort2vel = utils.KMM_velvort_to_velocity(grid)
vort2vel_fn = jax.jit(partial(im.vort2vel_newt, KMM_vort2vel_fn=KMMvort2vel))
vort2vel_fn_t = jax.vmap(vort2vel_fn)



real_traj_fn = partial(im.real_to_real_traj_fn_newt3d, vel2vort_fn=jax.vmap(vel2vort_fn), vort2vel_fn_t=jax.vmap(vort2vel_fn_t), traj_fn=jax.vmap(trajectory_fn)) 



# build model
newt_model = models.newt3d_sresol_unet(Nx_coarse, Ny_coarse, Nz_coarse,n_snapshots,
                                    N_filters=N_filters, N_layer=N_layer, N_deep=N_deep, N_grow=N_grow,
                                    kernel=kernel, input_channels=input_channels, output_channels=output_channels)

pooling_fn = jax.jit(im.coarse_pool_trajectory, static_argnums=(1, 2, 3))
pooling_fn_batched = jax.vmap(partial(pooling_fn, pool_d1=filter_size, pool_d2=filter_size, pool_d3=filter_size))


loss_fn = jax.jit(partial(lf.traj_VC_newt3d_sresol, 
                          trajectory_rollout_fn=real_traj_fn,
                          pooling_fn=pooling_fn_batched,
                          alpha=alphal, beta=betal,
                          vmax=0.05,
                          truth_weight_kind="uniform",
                          traj_weight_kind="exp_late",
                          weight_strength=1,))

newt_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_traj), loss=loss_fn, )



if (loadweights):
    print ('loading weights...')
    newt_model.load_weights(weight_loc+weight_name)

rng = np.random.default_rng(42)
Nwin = snapshots_trajs.shape[0]
perm = rng.permutation(Nwin)

train_idx = perm[:-nval]
val_idx   = perm[-nval:]

steps_per_epoch = int(np.ceil(train_idx.size / batch_size))
val_steps = int(np.ceil(val_idx.size / batch_size))


train_gen = keras_gen_sr_pretrain(
    snapshots_trajs, train_idx, batch_size,
    pooling_fn_batched=pooling_fn_batched,
    shuffle=True,
    rng=np.random.default_rng(0),)

val_gen = keras_gen_sr_pretrain(
    snapshots_trajs, val_idx, batch_size,
    pooling_fn_batched=pooling_fn_batched,
    shuffle=False,
    rng=None,)

hist_loss, hist_val_loss = [], []
for n in range(n_traj_steps):
  print("Traj step: ", n)

  history = newt_model.fit( train_gen,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=1, verbose=2, )

  current_loss = history.history["loss"][-1]
  current_val_loss = history.history['val_loss'][-1]

  hist_loss.append(current_loss)
  hist_val_loss.append(current_val_loss)

  if (not np.isfinite(current_loss)) or (not np.isfinite(current_val_loss)):
    print("Non-finite loss detected (loss or val_loss is NaN/Inf). Stopping training.")
    break

  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    newt_model.save_weights(weight_loc+weight_name)
