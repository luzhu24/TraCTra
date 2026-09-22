""" Attempt to convert old tf keras model to jax backend """
import tensorflow as tf
import keras
import keras.backend as K
import keras.ops as kops
from keras.layers import Conv2D, Lambda, Dense, Layer, TimeDistributed
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, GroupNormalization, Concatenate, Activation
from tensorflow.keras.initializers import Constant

from keras.layers import Conv3D
from keras.models import Model
import jax.numpy as jnp
import jax
import numpy as np





###################################### 3D periodic UNet (density -> u,v,w) ################################
#################################################################################

def pad_periodic_3d(x, n_pad_x=0, n_pad_y=0, n_pad_z=0):
  """
  Periodic padding for a 5D tensor (B, Nx, Ny, Nz, C) with channels-last.
  Pads along spatial axes x,y,z by (n_pad_x, n_pad_y, n_pad_z) total points.
  """
  def _pad_axis(t, axis, n_pad):
    if n_pad is None or int(n_pad) == 0:
      return t
    n_pad = int(n_pad)
    left = n_pad // 2
    right = n_pad - left

    parts = []
    if left > 0:
      slc = [slice(None)] * t.ndim
      slc[axis] = slice(-left, None)
      parts.append(t[tuple(slc)])
    parts.append(t)
    if right > 0:
      slc = [slice(None)] * t.ndim
      slc[axis] = slice(0, right)
      parts.append(t[tuple(slc)])

    return kops.concatenate(parts, axis=axis)

  x = _pad_axis(x, axis=1, n_pad=n_pad_x)  # Nx
  x = _pad_axis(x, axis=2, n_pad=n_pad_y)  # Ny
  x = _pad_axis(x, axis=3, n_pad=n_pad_z)  # Nz
  return x


def periodic_convolution_3d(x, n_filters, kernel, activation='relu',
                            strides=(1,1,1), n_pad_x=0, n_pad_y=0, n_pad_z=0,
                            kernel_initializer="glorot_uniform"):
  """
  Periodic-pad in (x,y,z) then apply Conv3D(valid).
  Expects channels-last tensors: (B, Nx, Ny, Nz, C).
  """
  padded_shape = (x.shape[1] + n_pad_x, x.shape[2] + n_pad_y, x.shape[3] + n_pad_z, x.shape[-1])

  x_padded = Lambda(
      pad_periodic_3d,
      arguments={'n_pad_x': n_pad_x, 'n_pad_y': n_pad_y, 'n_pad_z': n_pad_z},
      output_shape=padded_shape
  )(x)

  return Conv3D(n_filters, kernel, activation=activation, padding='valid', strides=strides,
      kernel_initializer=kernel_initializer)(x_padded)


def residual_block_periodic_conv_3d(x, n_filters,
                                       kernel=(3,3,3), strides=(1,1,1),
                                       n_pad_x=2, n_pad_y=2, n_pad_z=2,
                                       activation='gelu'):
  """
  3D version of residual_block_periodic_conv_LZ:
    BN -> act -> periodic_conv(linear)
    BN -> act -> periodic_conv(linear)
    + skip
  """
  layer_input = x

  # Project the residual branch only when the target number of channels changes.
  # This allows n_filters to vary across U-Net levels.
  if x.shape[-1] != n_filters:
    layer_input = Conv3D(
        n_filters, (1,1,1), padding='same', strides=strides,
        activation='linear')(layer_input)

  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = periodic_convolution_3d(x, n_filters, kernel, strides=strides,
      n_pad_x=n_pad_x, n_pad_y=n_pad_y, n_pad_z=n_pad_z,
      activation='linear')

  x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = periodic_convolution_3d(x, n_filters, kernel, strides=strides,
      n_pad_x=n_pad_x, n_pad_y=n_pad_y, n_pad_z=n_pad_z,
      activation='linear')

  x = keras.layers.add([x, layer_input])
  return x





def make_leray_projection_3d(Nx, Ny, Nz, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi, eps=1e-12):
    """
    Returns a JIT-able function proj(u) that projects u onto div-free fields.
    Assumes u is periodic in x,y,z and has shape (B, Nx, Ny, Nz, 3).
    """
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    # Wavenumbers (broadcast-friendly shapes)
    kx = (2*jnp.pi * jnp.fft.fftfreq(Nx, d=dx)).reshape(1, Nx, 1, 1)          # (1,Nx,1,1)
    ky = (2*jnp.pi * jnp.fft.fftfreq(Ny, d=dy)).reshape(1, 1, Ny, 1)          # (1,1,Ny,1)
    kz = (2*jnp.pi * jnp.fft.rfftfreq(Nz, d=dz)).reshape(1, 1, 1, Nz//2 + 1)  # (1,1,1,Nzr)

    k2 = kx**2 + ky**2 + kz**2
    inv_k2 = jnp.where(k2 > 0, 1.0 / (k2 + eps), 0.0)  # set k=0 mode safely

    @jax.jit
    def proj(u):
        if u.ndim != 5 or u.shape[-1] != 3:
            raise ValueError(f"Expected (B,Nx,Ny,Nz,3), got {u.shape}")

        # FFT in x,y,z (rfft along z)
        u_hat = jnp.fft.rfftn(u, axes=(1,2,3))  # (B,Nx,Ny,Nzr,3)
        ux, uy, uz = u_hat[..., 0], u_hat[..., 1], u_hat[..., 2]  # each (B,Nx,Ny,Nzr)

        # div_hat = i k · u_hat
        div_hat = 1j * (kx * ux + ky * uy + kz * uz)

        # Solve Poisson: Δφ = div(u)  =>  φ_hat = - div_hat / |k|^2
        phi_hat = -div_hat * inv_k2

        # Correction = -∇φ  =>  corr_hat = - i k φ_hat
        corr_hat = -1j * jnp.stack([kx * phi_hat, ky * phi_hat, kz * phi_hat], axis=-1)

        corr = jnp.fft.irfftn(corr_hat, s=(Nx, Ny, Nz), axes=(1,2,3))  # back to real space
        return u + corr

    return proj


def div_free_3D_layer(u_in):
    """
    u_in: Keras tensor with shape (None, Nx, Ny, Nz, 3)
    """
    if len(u_in.shape) != 5:
        raise ValueError("Expected 5D input (B,Nx,Ny,Nz,3), got", u_in.shape)
    if u_in.shape[-1] != 3:
        raise ValueError("Expected 3 channels (u,v,w), got", u_in.shape[-1])

    _,Nx,Ny,Nz,_ = u_in.shape

    proj3d = make_leray_projection_3d(Nx,Ny,Nz)  # captures precomputed k-grids
    return Lambda(lambda x: proj3d(x), output_shape=u_in.shape[1:])(u_in)



class AveragingBlurInitializer3D(keras.initializers.Initializer):
  def __call__(self, shape, dtype=None):
    # shape = (kx, ky, kz, in_ch, out_ch) for Conv3D
    if dtype is None:
      dtype = tf.float32
    kx, ky, kz, in_ch, out_ch = shape
    k_area = float(kx * ky * kz)
    base_kernel = np.ones((kx, ky, kz), dtype=np.float32) / k_area

    w = np.zeros(shape, dtype=np.float32)
    channels = min(in_ch, out_ch)
    for c in range(channels):
      w[:, :, :, c, c] = base_kernel
    return tf.convert_to_tensor(w, dtype=dtype)

  def get_config(self):
    return {}



def newt3d_single_sresol_unet(
    Nx_coarse, Ny_coarse, Nz_coarse,
    N_filters=32,
    N_layer=2,
    N_deep=4,
    N_grow=3,
    filter_factor=2,
    kernel=(3,3,3),
    input_channels=1,
    output_channels=3,
):
  """
  3D periodic ResUNet:
    input : (Nxcoarse, Nycoarse, Nzcoarse, Cin)  
    output: (Nx, Ny, Nz, Cout) 
  """

  inp = Input(shape=(Nx_coarse, Ny_coarse, Nz_coarse, input_channels), name="frame_input")

  up_factor = 2 ** N_grow

  kx, ky, kz = kernel
  pad_x, pad_y, pad_z = kx - 1, ky - 1, kz - 1

  # Pre-upsampling
  x = tf.keras.layers.UpSampling3D(
      size=(up_factor, up_factor, up_factor),
      name="coarse_to_hr_bilinear",)(inp)

  # ---- Stem ----
  x = periodic_convolution_3d(x, N_filters, kernel=kernel,
                          n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, activation="linear" )

  # ---- Encoder ----
  skips = []
  filter_schedule = [
      int(round(N_filters * (filter_factor ** lev)))
      for lev in range(N_deep)
  ]

  for lev in range(N_deep):
    filters = filter_schedule[lev]

    # Local processing at this scale
    for _ in range(max(1, N_layer)):
      x = residual_block_periodic_conv_3d(x, filters, kernel=kernel,
                                  n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, )
    skips.append(x)

    if lev < N_deep - 1:
      x = tf.keras.layers.AveragePooling3D(pool_size=2)(x)

  # ---- Bottleneck ----
  for _ in range(max(1, N_layer)):
    x = residual_block_periodic_conv_3d(x, filters, kernel=kernel,
                                    n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, )

  # ---- Decoder ----
  for lev in reversed(range(N_deep - 1)):
    filters = filter_schedule[lev]

    x = tf.keras.layers.UpSampling3D(size=2)(x)

    x = periodic_convolution_3d(
        x, filters, kernel=kernel,
        n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z,
        activation="gelu" )

    x = Concatenate(axis=-1)([x, skips[lev]])

    for _ in range(N_layer):
      x = residual_block_periodic_conv_3d(x, filters, kernel=kernel,
                                      n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, )

  last_init = AveragingBlurInitializer3D()

  # ---- Head ----
  uvw_raw = periodic_convolution_3d(x, output_channels, kernel=kernel,
                              n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z,
                              activation="linear", kernel_initializer=last_init )

  out = div_free_3D_layer(uvw_raw)

  return keras.Model(inp, out, name="newt3d_sresol_single_unet")


def newt3d_sresol_unet(
    Nx_coarse, Ny_coarse, Nz_coarse,Nt,
    N_filters=32,
    N_layer=2,
    N_deep=4,
    N_grow=3,
    filter_factor=2,
    kernel=(3,3,3),
    input_channels=1,
    output_channels=3):
  """
  Apply the 3D volume U-Net to each time in a trajectory:
    input : (Nt, Nx, Ny, Nz, Cin)
    output: (Nt, Nx, Ny, Nz, Cout)
  """
  inp = Input(shape=(Nt, Nx_coarse, Ny_coarse, Nz_coarse, input_channels), name="traj_input")

  vol_model = newt3d_single_sresol_unet(Nx_coarse, Ny_coarse, Nz_coarse,
      N_filters=N_filters, N_layer=N_layer, N_deep=N_deep, N_grow=N_grow,
      filter_factor=filter_factor, kernel=kernel,
      input_channels=input_channels, output_channels=output_channels,)

  out = TimeDistributed(vol_model, name="per_t_3d_unet")(inp)
  return keras.Model(inp, out, name="newt3d_sresol_unet")
