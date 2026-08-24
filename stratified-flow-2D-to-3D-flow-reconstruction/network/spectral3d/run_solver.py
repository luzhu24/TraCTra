import numpy as np
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import os.path

import time_forward_maps as tfm
import initial_condition as ic
from transform_traj import *

jax.config.update("jax_enable_x64", True)
print("Run on ", jax.lib.xla_bridge.get_backend().platform)


# configure
init_from_lam = 1  # old stuff, is currently inactive ...
write_traj = False # -> if False, we write each uvw snapshot into a separate file
#write_traj = True # -> if True, we write the full uvw trajectories into separate files

file_front = "/mnt/ceph_rbd/markus-temp/result_N128_snaps/"

forcing = "NonhelicalSinusoidal"
forcedir = 0  # for NonhelicalSinusoidal, this forcing is only a dummy
kf = 3        # for NonhelicalSinusoidal, this mean all k <= kf are forced

Re = 50. 
visc = 1. / Re

alpha = 1. # domain in x = alpha * L
beta  = 1. # domain in y = beta * L
gamma = 1. # domain in z = gamma * L
L = 2 * jnp.pi 
Lx = alpha * L
Ly = alpha * L
Lz = alpha * L

Nx = 128
Ny = Nx
Nz = Nx

offsets = ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.))
grid = cfd.grids.Grid((Nx, Ny, Nz), domain=((0., Lx), (0., Ly), (0., Lz)))
x, y, z = grid.mesh(offsets[0])


# simulation setup
simulation_config = tfm.SimulationConfig(
                        grid=grid,
                        visc=visc,
                        kf=kf,
                        forcing=forcing,
                        forcedir=forcedir)

N_downsample = None  # downsample field for write out? 
N_traj = 75         # how many trajectories 
T_burn = 100.        # length of burn in interval
T_write_out = 100    # length of trajectory (after burn in)
dt_write_out =  2    # time interval to write data


# generate time forward maps
forward_map_burn = simulation_config.generate_time_forward_map(
                                     int(T_burn / simulation_config.dt_stable), 
                                     simulation_config.dt_stable)
trajectory_fn = simulation_config.generate_trajectory_fn(T_write_out, dt_write_out)



def write_out_uvwtrajectory(
    vx: jnp.ndarray, 
    vy: jnp.ndarray, 
    vz: jnp.ndarray, 
    file_front: str, 
    file_number: int, 
    n_zeros=3,
    write_traj=True
  ):
  #
  vx_array = jnp.array(vx)
  vy_array = jnp.array(vy)
  vz_array = jnp.array(vz)
  Nsnaps = vx_array.shape[0]
  #
  if write_traj:  
    while True: # check which fields exist
      file_name = str(file_front) + "uvw_Re" + str(round(Re, 2))  + "_traj." + str(file_number).zfill(n_zeros) + ".npy"
      if os.path.exists(file_name):
        file_number += 1 
        continue
      else:
        break
    #
    with open(file_name, 'wb') as f:
      np.save(f, vx_array)
      np.save(f, vy_array)
      np.save(f, vz_array)
      
  else:
    for isnap in range(0,Nsnaps):
      if isnap==0:
        while True: # check which fields exist
          file_name = str(file_front) + "uvw_Re" + str(round(Re, 2))  + "_traj." + str(file_number).zfill(n_zeros) \
                                      + "_snap" + str(isnap).zfill(n_zeros) + ".npy"
          if os.path.exists(file_name):
            file_number += 1 
            continue
          else:
            break
        #
      else:
        file_name = str(file_front) + "uvw_Re" + str(round(Re, 2))  + "_traj." + str(file_number).zfill(n_zeros) \
                                    + "_snap" + str(isnap).zfill(n_zeros) + ".npy"
              
      with open(file_name, 'wb') as f:
        dat = jnp.concatenate([vx_array[isnap, ..., np.newaxis],\
                               vy_array[isnap, ..., np.newaxis],\
                               vz_array[isnap, ..., np.newaxis]], axis=-1)  
        np.save(f, dat) 
          
  return


# generate trajectories
for traj_num in range(N_traj):
   # generate initial condition
   #max_vel = 0.5 * Re / kf ** 2
   max_vel = 5
   xvec_initial_rft  = ic.set_initField(grid=grid, init_from_lam=init_from_lam, max_vel=max_vel, seed=None)
   xvec_burnedin_rft = forward_map_burn(xvec_initial_rft)
   xvec_final_rft, xvec_traj_rft = trajectory_fn(xvec_burnedin_rft)

   #if N_downsample is not None:
   #  print("Downsampling not yet implemented!")
   #  xvec_traj_rft = downsample_rft(xvec_traj_rft, N_downsample)

   vx, vy, vz = rft_trajectory_to_physical(xvec_traj_rft, grid)
   write_out_uvwtrajectory(vx, vy, vz, file_front, traj_num, n_zeros=4, write_traj=write_traj)

