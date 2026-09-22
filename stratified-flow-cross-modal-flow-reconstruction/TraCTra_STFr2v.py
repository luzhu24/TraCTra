import os
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
from network import sym_augment as sa

from network.spectral3d import utils


# Use parameters from config file
def loadnpz(fieldname,filename='./data.npz'):
    fields=[]
    dnsdata = np.load(f'{filename}')
    for fn in fieldname:
        fields.append(np.asarray(dnsdata[fn],dtype=np.float32))
    param = dnsdata['param']
    geom = dnsdata['geom']
    fields = np.array(fields).transpose((1,2,3,4,0))
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




data_loc = '../data/training_data/'
weight_loc = './'
file_front = 'data_re500_cubic'
file_end = '.npz'
n_files = 200

loadweights = False
weight_name = 'stf3d_best_traj_VC.test.weights.h5'
snapshots,param,geom = loadnpz(['vx','vy','vz','r'],data_loc+file_front+file_end)
re,pr,ri,theta,kf,sbk = param
Nx,Ny,Nz,Lx,Ly,Lz,Nt,dt_dns = geom
Nx,Ny,Nz = int(Nx),int(Ny),int(Nz)
smooth = False
print (geom,param,snapshots.shape)

noise_level = 0.1

# network hyp
n_input = 1
n_output = 3
T_unroll = 10.
T_interval = 1
stride = 1
dt_stable = 0.02 
V_weight,C_weight=1,1

# training hyp
batch_size = 1
lr_traj = 5e-5#
nval = 4
n_traj_steps = 1000
n_layer = 2
n_level = 3
kernel = (3,3,3)
n_filters=16
filter_factor=1.5

grid = cfd.grids.Grid((Nx, Ny, Nz), domain=((0, Lx), (-Ly/2., Ly/2.), (-Lz/2., Lz/2.)))

max_vel_est = 1.

snapshots_trajs,n_snapshots = make_subtrajectories(snapshots, T_unroll, dt_dns, stride=stride, T_interval=T_interval)
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
stf3d_model = models.density_to_uvw_traj_unet_3d(
    Nx, Ny, Nz, n_snapshots,
    N_filters=n_filters, N_layer=n_layer, N_levels=n_level, kernel=kernel,
    input_channels=n_input, output_channels=n_output,
    filter_factor=filter_factor)


print("\n================ Model Summary ================\n")
stf3d_model.summary()


def print_model_layers(model, indent=0):
    """Print layer-by-layer information, including nested models."""
    prefix = " " * indent
    for i, layer in enumerate(model.layers):
        try:
            input_shape = layer.input.shape
        except (AttributeError, ValueError):
            input_shape = "N/A"

        try:
            output_shape = layer.output.shape
        except (AttributeError, ValueError):
            output_shape = "N/A"

        print(
            f"{prefix}[{i:02d}] {layer.name:<30s} "
            f"{layer.__class__.__name__:<20s} "
            f"input={input_shape} output={output_shape} "
            f"params={layer.count_params()}"
        )

        sublayer = getattr(layer, "layer", None)
        if isinstance(sublayer, keras.Model):
            print(f"{prefix}     -> nested model: {sublayer.name}")
            print_model_layers(sublayer, indent=indent + 8)
        elif isinstance(layer, keras.Model):
            print_model_layers(layer, indent=indent + 8)


print("\n================ Layer Information ================\n")
print_model_layers(stf3d_model)
print("\n===================================================\n")


loss_fn = jax.jit(partial(lf.traj_VC_stf3d_r2v, trajectory_rollout_fn=real_traj_fn, alpha=V_weight, beta=C_weight, vmax=5))

stf3d_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_traj), loss=loss_fn, metrics=[keras.losses.MeanSquaredError()])



if (loadweights):
    print ('loading weights...')
    stf3d_model.load_weights(weight_loc+weight_name)


def keras_gen(arr, idx, batch_size, n_input, *, shuffle=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    while True:                       
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, order.size, batch_size):
            b = order[s:s+batch_size]                

            batch = arr[b]
            x = batch[..., -n_input:]              
            y = batch[..., -n_input:]                 
            yield x, y

rng = np.random.default_rng(42)
Nwin = snapshots_trajs.shape[0]
perm = rng.permutation(Nwin)

train_idx = perm[:-nval]
val_idx   = perm[-nval:]

steps_per_epoch = int(np.ceil(train_idx.size / batch_size))
val_steps = int(np.ceil(val_idx.size / batch_size))

train_gen = keras_gen(snapshots_trajs, train_idx, batch_size, n_input, shuffle=False, rng=np.random.default_rng(0))
val_gen   = keras_gen(snapshots_trajs,   val_idx, batch_size, n_input, shuffle=False)



hist_loss, hist_val_loss = [], []
for n in range(n_traj_steps):
  print("Traj step: ", n)

  history = stf3d_model.fit( train_gen,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=1, verbose=2,
                            )

  current_loss = history.history["loss"][-1]
  current_val_loss = history.history['val_loss'][-1]

  hist_loss.append(current_loss)
  hist_val_loss.append(current_val_loss)

  # check for NaN/Inf on either
  if (not np.isfinite(current_loss)) or (not np.isfinite(current_val_loss)):
    print("Non-finite loss detected (loss or val_loss is NaN/Inf). Stopping training.")
    break

  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    stf3d_model.save_weights(weight_loc+weight_name)


