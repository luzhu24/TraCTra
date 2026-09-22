""" Attempt to convert old tf keras model to jax backend """
import tensorflow as tf
import keras
import keras.ops as kops
from keras.layers import Conv2D, Conv3D, Lambda, TimeDistributed, Reshape
from keras.layers import Input
from keras.layers import BatchNormalization, Concatenate, Activation

import jax.numpy as jnp
import jax
import numpy as np

def pad_periodic(x, n_pad_rows=0, n_pad_cols=0):
    """Periodic pad for 4D tensor (B, Nx, Ny, C), channels-last."""
    if n_pad_rows is None:
        n_pad_rows = 0
    if n_pad_cols is None:
        n_pad_cols = 0

    n_pad_rows = int(n_pad_rows)
    n_pad_cols = int(n_pad_cols)

    # pad rows
    if n_pad_rows > 0:
        top = n_pad_rows // 2
        bottom = n_pad_rows - top

        parts = []
        if top > 0:
            parts.append(x[:, -top:, :, :])
        parts.append(x)
        if bottom > 0:
            parts.append(x[:, :bottom, :, :])

        x = kops.concatenate(parts, axis=1)

    # pad cols
    if n_pad_cols > 0:
        left = n_pad_cols // 2
        right = n_pad_cols - left

        parts = []
        if left > 0:
            parts.append(x[:, :, -left:, :])
        parts.append(x)
        if right > 0:
            parts.append(x[:, :, :right, :])

        x = kops.concatenate(parts, axis=2)

    return x


def periodic_convolution(x, n_filters, kernel, activation='relu',
    strides=(1,1), n_pad_rows=0, n_pad_cols=0, kernel_initializer="glorot_uniform"):
    """2D periodic convolution."""


    if int(n_pad_rows) == 0 and int(n_pad_cols) == 0:
        return Conv2D(n_filters, kernel, activation=activation, padding='valid',
            strides=strides, kernel_initializer=kernel_initializer)(x)

    padded_shape = (x.shape[1] + n_pad_rows, x.shape[2] + n_pad_cols, x.shape[-1])

    x_padded = Lambda(pad_periodic, arguments={'n_pad_rows': n_pad_rows, 'n_pad_cols': n_pad_cols},
        output_shape=padded_shape )(x)

    return Conv2D(n_filters, kernel, activation=activation, padding='valid',
        strides=strides, kernel_initializer=kernel_initializer)(x_padded)



def residual_block_periodic_conv_LZ(x, n_filters,
                                 kernel=(1,1), strides=(1,1),
                                 n_pad_rows=1, n_pad_cols=1,
                                 activation='gelu'):
  layer_input = x

  # Project the residual shortcut only when the channel count changes.
  if x.shape[-1] != n_filters:
    layer_input = Conv2D(n_filters, (1,1), padding='same', strides=strides, activation='linear')(layer_input)

  x = BatchNormalization()(x)
  x = Activation(activation)(x)

  x = periodic_convolution(x, n_filters, kernel, strides=strides,
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')
  
  x = BatchNormalization()(x)
  x = Activation(activation)(x)

  x = periodic_convolution(x, n_filters, kernel, strides=strides,
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')

  x = keras.layers.add([x, layer_input])
  return x






###################################### Stratification --LZ ################################
#################################################################################

class AveragingBlurInitializer(keras.initializers.Initializer):
  def __call__(self, shape, dtype=None):
    # shape = (kh, kw, in_ch, out_ch) for Conv2D
    if dtype is None:
      dtype = tf.float32

    kh, kw, in_ch, out_ch = shape
    k_area = float(kh * kw)
    base_kernel = np.ones((kh, kw), dtype=np.float32) / k_area

    w = np.zeros(shape, dtype=np.float32)
    channels = min(in_ch, out_ch)
    for c in range(channels):
      w[:, :, c, c] = base_kernel  # blur input channel c into output channel c

    return tf.convert_to_tensor(w, dtype=dtype)

  def get_config(self):
    return {}






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
  # Project the residual shortcut only when the channel count changes.
  if x.shape[-1] != n_filters:
    layer_input = Conv3D(n_filters, (1,1,1), padding='same', strides=strides, activation='linear')(layer_input)

  x = BatchNormalization()(x)#GN(x)#
  x = Activation(activation)(x)
  x = periodic_convolution_3d(x, n_filters, kernel, strides=strides,
      n_pad_x=n_pad_x, n_pad_y=n_pad_y, n_pad_z=n_pad_z,
      activation='linear')

  x = BatchNormalization()(x)#GN(x)#
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




######################## 2D-> 3D #####################################

def learned_lift_2d_to_3d(x, nz, n_filters, *, #rank=8,
    activation="gelu", name="lift2d3d",):
  """
  Learned 2D -> 3D lift.

  Input:
      x  : (B, Nx, Ny, C)
  Output:
      out: (B, Nx, Ny, Nz, n_filters)
  """
  nx0 = int(x.shape[1])
  ny0 = int(x.shape[2])


  # learned per-(x,y) projection into z-channels
  x = periodic_convolution(x, nz * n_filters, kernel=(1, 1), 
      activation="linear", n_pad_rows=0, n_pad_cols=0,)
  

  x = Reshape((nx0, ny0, nz, n_filters), name=f"{name}_reshape")(x)

  # light 3D mixing after lift
  x = periodic_convolution_3d(x, n_filters, kernel=(1, 1, 3),
      activation=activation, n_pad_x=0, n_pad_y=0, n_pad_z=2,)
  
  return x


def slice_to_uvwr_single_unet_lift3d(
    Nx, Ny, Nz,
    N_filters=32,
    N_layer=2,
    N_levels=3,
    kernel2d=(3,3),
    kernel3d=(3,3,3),
    input_channels=1,
    velocity_channels=3,
    density_channels=1,
    filter_factor=2,
):
  """
  2D density slice -> full 3D (u,v,w,rho)

  Input:
      (Nx, Ny, Cin)
  Output:
      (Nx, Ny, Nz, 3 + density_channels)
  """
  fac = 2 ** (N_levels - 1)
  if Nx % fac != 0 or Ny % fac != 0 or Nz % fac != 0:
    raise ValueError(
        f"Nx, Ny, Nz must be divisible by 2**(N_levels-1)={fac}. "
        f"Got {(Nx, Ny, Nz)}"
    )

  inp = Input(shape=(Nx, Ny, input_channels), name="slice_input")

  kx2, ky2 = kernel2d
  pad_rows, pad_cols = kx2 - 1, ky2 - 1

  kx3, ky3, kz3 = kernel3d
  pad_x, pad_y, pad_z = kx3 - 1, ky3 - 1, kz3 - 1

  # 2D encoder
  x = periodic_convolution(inp, N_filters, kernel=kernel2d, n_pad_rows=pad_rows, n_pad_cols=pad_cols, activation="linear" )

  skips2d = []
  filter_schedule = [int(round(N_filters * (filter_factor ** lev))) for lev in range(N_levels)]

  for lev in range(N_levels):
    filters = filter_schedule[lev]
    for _ in range(N_layer):
      x = residual_block_periodic_conv_LZ(x, filters, kernel=kernel2d, n_pad_rows=pad_rows, n_pad_cols=pad_cols, )

    skips2d.append(x)

    if lev < N_levels - 1:
      x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

  # 2D bottleneck refinement
  filters = filter_schedule[-1]
  for _ in range(max(1, N_layer)):
    x = residual_block_periodic_conv_LZ(x, filters, kernel=kernel2d, n_pad_rows=pad_rows, n_pad_cols=pad_cols, )

  # learned lift at bottleneck
  nz_bottleneck = Nz // fac

  x = learned_lift_2d_to_3d(x, nz=nz_bottleneck, n_filters=filters, activation="gelu", name="bottleneck_lift", )

  for _ in range(max(1, N_layer)):
    x = residual_block_periodic_conv_3d(x, filters, kernel=kernel3d,
      n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, )

  # 3D decoder with learned lifted skips
  for lev in reversed(range(N_levels - 1)):
    filters = filter_schedule[lev]

    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = periodic_convolution_3d(x, filters, kernel=kernel3d, n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, activation="gelu" )

    nz_here = Nz // (2 ** lev)

    skip3d = learned_lift_2d_to_3d(skips2d[lev], nz=nz_here, n_filters=filters, activation="gelu", name=f"skip_lift_l{lev}",)

    x = Concatenate(axis=-1)([x, skip3d])

    # compress after concat so channel count stays controlled
    x = periodic_convolution_3d(x, filters, kernel=(1,1,1), n_pad_x=0, n_pad_y=0, n_pad_z=0, activation="gelu" )

    for _ in range(N_layer):
      x = residual_block_periodic_conv_3d(x, filters, kernel=kernel3d,
          n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z, )

  # head: predict uvw + rho
  last_init = AveragingBlurInitializer3D()

  raw = periodic_convolution_3d(
      x, velocity_channels + density_channels, kernel=kernel3d,
      n_pad_x=pad_x, n_pad_y=pad_y, n_pad_z=pad_z,
      activation="linear", kernel_initializer=last_init)
  

  uvw_raw = Lambda(lambda t: t[..., :velocity_channels], name="uvw_raw")(raw)
  rho_out = Lambda(lambda t: t[..., velocity_channels:], name="rho_out")(raw)

  uvw_out = div_free_3D_layer(uvw_raw)

  out = Concatenate(axis=-1, name="uvw_rho_out")([uvw_out, rho_out])

  return keras.Model(inp, out,name="slice_to_uvwr_single_unet_lift3d"  )


def slice_to_uvwr_traj_unet_lift3d(
    Nx, Ny, Nz, Nt,
    N_filters=32,
    N_layer=2,
    N_levels=3,
    kernel2d=(3,3),
    kernel3d=(3,3,3),
    input_channels=1,
    velocity_channels=3,
    density_channels=1,
    filter_factor=2,
):
  """
  Apply the 2D-slice -> 3D-volume model at each time step.

  Input:
      (Nt, Nx, Ny, Cin)
  Output:
      (Nt, Nx, Ny, Nz, 3 + density_channels)
  """
  inp = Input(shape=(Nt, Nx, Ny, input_channels), name="slice_traj_input")

  frame_model = slice_to_uvwr_single_unet_lift3d(Nx, Ny, Nz,
      N_filters=N_filters, N_layer=N_layer, N_levels=N_levels,
      kernel2d=kernel2d, kernel3d=kernel3d,
      input_channels=input_channels,
      velocity_channels=velocity_channels,
      density_channels=density_channels,
      filter_factor=filter_factor, )

  out = TimeDistributed(frame_model, name="per_t_2d_to_3d")(inp)
  return keras.Model(inp, out, name="slice_to_uvwr_traj_unet_lift3d")
