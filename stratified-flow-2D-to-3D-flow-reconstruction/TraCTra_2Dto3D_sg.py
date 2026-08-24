
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


def loadnpz(fieldname,filename='./data.npz'):
    fields=[]
    dnsdata = np.load(f'{filename}')
    for fn in fieldname:
        fields.append(np.asarray(dnsdata[fn],dtype=np.float32))#
    param = dnsdata['param']
    geom = dnsdata['geom']
    fields = np.array(fields).transpose((1,2,3,4,0))
    return fields,param,geom



def make_subtrajectories(A, dT, dt_dns, offset=0, stride=1, T_interval=1):

    A = np.asarray(A)
    T = A.shape[0]
    L = int(np.floor(dT / dt_dns))
    if T < L:
        raise ValueError(f"Not enough time frames: T={T} < L={L}")

    W = sliding_window_view(A, window_shape=L+2*offset, axis=0)  
    W = np.moveaxis(W, -1, 1)                              
    W = W[::stride, ::T_interval]                            
    return W, W.shape[1]-2*offset



def slicing_traj(m, ax=1, slice_channel=-1,
    Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi, normalise=False,):

    def _kvec(n, L):
        return 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)

    # density field: (nt, nx, ny, nz)
    rho = m[..., slice_channel]
    print (rho.shape)
    nx, ny, nz = rho.shape

    kx = _kvec(nx, Lx).reshape(nx, 1, 1)
    ky = _kvec(ny, Ly).reshape(1, ny, 1)
    kz = _kvec(nz, Lz).reshape(1, 1, nz)

    rho_hat = jnp.fft.fftn(rho, axes=(0, 1, 2))

    if ax == 0:          # view along x -> integrate over x, keep (y,z)
        lap_perp_hat = -(ky**2 + kz**2) * rho_hat
        lap_perp = jnp.fft.ifftn(lap_perp_hat, axes=(0, 1, 2)).real
        shadow = jnp.sum(lap_perp, axis=0) * (Lx / nx)

    elif ax == 1:        # view along y -> integrate over y, keep (x,z)
        lap_perp_hat = -(kx**2 + kz**2) * rho_hat
        lap_perp = jnp.fft.ifftn(lap_perp_hat, axes=(0, 1, 2)).real
        shadow = jnp.sum(lap_perp, axis=1) * (Ly / ny)

    elif ax == 2:        # view along z -> integrate over z, keep (x,y)
        lap_perp_hat = -(kx**2 + ky**2) * rho_hat
        lap_perp = jnp.fft.ifftn(lap_perp_hat, axes=(0, 1, 2)).real
        shadow = jnp.sum(lap_perp, axis=2) * (Lz / nz)

    else:
        raise ValueError("ax must be 0, 1, and 2 for x, y, or z line-of-sight.")

    if normalise:
        smax = jnp.max(jnp.abs(shadow), axis=tuple(range(1, shadow.ndim)), keepdims=True)
        shadow = shadow / (smax + 1e-12)
    

    return shadow[...,None]



data_loc = '/home/zhulu/jax-cfd/flow3d/DNS/'#'/mnt/ceph_rbd/flow3d/DNS/'
weight_loc = './'#'/mnt/ceph_rbd/flow3d/stratified3d_2dto3d_re500/'
file_front = 'data_re500_T600N300'
file_end = '.npz'
n_files = 200

loadweights = False
weight_name = 'stf3d_best_slice_R1V1_T600.sg.weights.h5'
snapshots,param,geom = loadnpz(['vx','vy','vz','r'],data_loc+file_front+file_end)
re,pr,ri,theta,kf,sbk = param
Nx,Ny,Nz,Lx,Ly,Lz,Nt,dt_dns = geom
Nx,Ny,Nz = int(Nx),int(Ny),int(Nz)
smooth = False


print (geom,param,snapshots.shape)

noise_level = 0.1

# network hyp
channels=[-1]
input_channels=len(channels)
velocity_channels=3
density_channels=1
T_unroll = 16.
T_interval = 1
stride = 1
dt_stable = 0.2 
t_offset = 2
V_weight,C_weight=1,1


# training hyp
batch_size = 2
lr_traj = 5e-5
nval = 4
n_traj_steps = 2000
n_layer = 2
n_level = 3
kernel2d = (3,3)
kernel3d = (3,3,3)
n_filters=16


grid = cfd.grids.Grid((Nx, Ny, Nz), domain=((0, Lx), (-Ly/2., Ly/2.), (-Lz/2., Lz/2.)))

max_vel_est = 1.


slicing_fn = jax.jit(partial(slicing_traj, ax=2, slice_channel=channels[0], Lx=Lx, Ly=Ly, Lz=Lz, normalise=False))
slicing_fn_t = jax.vmap(slicing_fn)
snapshots = slicing_fn_t(snapshots)

snapshots_trajs,n_snapshots = make_subtrajectories(snapshots, T_unroll, dt_dns, offset=t_offset, stride=stride, T_interval=T_interval)
print (snapshots_trajs.shape,n_snapshots)

min_val_loss = np.inf

t_substep = (dt_dns*T_interval)/dt_stable
N_substep = t_substep*n_snapshots
print("dt_stable:", dt_stable, "T substep:", t_substep, "N_substep:", N_substep)
trajectory_fn = ts.generate_trajectory_fn_stf3d(re,pr,ri,theta,kf,sbk, dt_stable, grid, t_substep=t_substep, n_substep=n_snapshots, smooth=smooth)


KMMvel2vort = utils.KMM_velocity_to_velvort(grid)
vel2vort_fn = jax.jit(partial(im.vel2vort, KMM_vel2vort_fn=KMMvel2vort))

KMMvort2vel = utils.KMM_velvort_to_velocity(grid)
vort2vel_fn = jax.jit(partial(im.vort2vel, KMM_vort2vel_fn=KMMvort2vel))
vort2vel_fn_t = jax.vmap(vort2vel_fn)


real_traj_fn = partial(im.real_to_real_traj_fn_stf3d, 
                       vel2vort_fn=jax.vmap(vel2vort_fn), vort2vel_fn_t=jax.vmap(vort2vel_fn_t), 
                       traj_fn=jax.vmap(trajectory_fn)) 



# build model 
stf3d_model = models.slice_to_uvwr_traj_unet_lift3d(Nx, Ny, Nz, n_snapshots, N_filters=n_filters, N_layer=n_layer, N_levels=n_level, kernel2d=kernel2d, kernel3d=kernel3d, input_channels=input_channels*(1+2*t_offset), velocity_channels=velocity_channels, density_channels=density_channels,)




loss_fn = jax.jit(partial(lf.traj_VC_stf3d_slice2vr, 
                          trajectory_rollout_fn=real_traj_fn,
                          slicing_fn=jax.vmap(slicing_fn_t),
                          alpha=V_weight,beta=C_weight,vmax=2,rmax=5,
                          truth_weight_kind="exp_late",
                          traj_weight_kind = "exp_late",
                          weight_strength = 1.0,))



stf3d_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_traj), loss=loss_fn, )



if (loadweights):
    print ('loading weights...')
    stf3d_model.load_weights(weight_loc+weight_name)


def keras_gen_offset(arr, idx, batch_size, *, offset=0, shuffle=True, rng=None):
    """
    arr: (Nwin, Nt_ext, Nx, Ny, Nz, Cin)

    yields
      x: (B, Nt, Nx, Ny, Nz, Cin*(2*offset+1))
      y: (B, Nt, Nx, Ny, Nz, Cin)   
    """
    if rng is None:
        rng = np.random.default_rng()

    Nt_ext = arr.shape[1]
    window = 2 * offset + 1
    Nt = Nt_ext - 2 * offset

    while True:
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, order.size, batch_size):
            b = order[s:s+batch_size]
            batch = arr[b]   # (B, Nt_ext, Nx, Ny, Nz, Cin)

            if offset == 0:
                x = batch
                y = batch
            else:
                parts = [batch[:, i:i+Nt, ...] for i in range(window)]
                x = np.concatenate(parts, axis=-1)          # stack time into channels
                y = batch[:, offset:offset+Nt, ...]         # center targets

            yield x, y

rng = np.random.default_rng(42)
Nwin = snapshots_trajs.shape[0]
perm = rng.permutation(Nwin)

train_idx = perm[:-nval]
val_idx   = perm[-nval:]

steps_per_epoch = int(np.ceil(train_idx.size / batch_size))
val_steps = int(np.ceil(val_idx.size / batch_size))

train_gen = keras_gen_offset(snapshots_trajs,train_idx, batch_size, offset=t_offset, shuffle=True, rng=np.random.default_rng(0))
val_gen   = keras_gen_offset(snapshots_trajs,  val_idx, batch_size, offset=t_offset, shuffle=False)



hist_loss, hist_val_loss = [], []
for n in range(n_traj_steps):
  print("Traj step: ", n)

  history = stf3d_model.fit( train_gen,
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
    stf3d_model.save_weights(weight_loc+weight_name)

np.savez(data_loc + 'history.npz', loss=np.array(hist_loss), val_loss=np.array(hist_val_loss))
