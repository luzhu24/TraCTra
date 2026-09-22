""" Attempt to convert old tf keras model to jax backend """
import tensorflow as tf
import keras
import keras.ops as kops
from keras.layers import Conv2D, Lambda, Dense, Layer, TimeDistributed
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Concatenate, Activation
from tensorflow.keras.initializers import Constant


from keras.models import Model
import jax.numpy as jnp
import numpy as np

def pad_periodic(x, n_pad_rows=0, n_pad_cols=0):
  """
  Pads the rows and columns of a 4D tensor in a periodic manner.
  """
  top_rows = x[:, -n_pad_rows // 2:, :, :]
  bottom_rows = x[:, :n_pad_rows // 2, :, :]
  padded_rows_x = kops.concatenate([top_rows, x, bottom_rows], axis=1)

  left_cols = padded_rows_x[:, :, -n_pad_cols // 2:, :]
  right_cols = padded_rows_x[:, :, :n_pad_cols // 2, :]
  padded_rows_cols_x = kops.concatenate([left_cols, padded_rows_x, right_cols], axis=2)
  return padded_rows_cols_x


def periodic_convolution(x, n_filters, kernel, activation='relu', 
                         strides=(1,1), n_pad_rows=0, n_pad_cols=0,
                         kernel_initializer="glorot_uniform"):
  """ 
  Applies periodic boundary conditions before convolving an input tensor with a set of filters.
  """
  # shape computation with padding; does not include batch dimension 
  padded_shape = (x.shape[1] + n_pad_rows, x.shape[2] + n_pad_cols, x.shape[-1])

  x_padded = Lambda(pad_periodic, arguments={'n_pad_rows': n_pad_rows, 'n_pad_cols': n_pad_cols},
                    output_shape=padded_shape)(x)
  return Conv2D(n_filters, kernel, activation=activation, padding='valid', strides=strides,kernel_initializer=kernel_initializer)(x_padded)


def residual_block_periodic_conv_LZ(x, n_filters,
                                 kernel=(1,1), strides=(1,1),
                                 n_pad_rows=1, n_pad_cols=1,
                                 activation='gelu'):
  layer_input = x

  # Project the residual branch when the number of channels changes.
  # This allows n_filters to follow the U-Net level instead of being
  # forced to x.shape[-1].
  if x.shape[-1] != n_filters:
    layer_input = Conv2D(n_filters, (1,1), padding='same',
                         strides=strides, activation='linear')(layer_input)

  x = BatchNormalization()(x)
  x = Activation(activation)(x)

  x = periodic_convolution(x, n_filters, kernel, strides=strides,
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')
  
  x = BatchNormalization()(x)
  x = Activation(activation)(x)

  x = periodic_convolution(x, n_filters, kernel, strides=(1,1),
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')

  x = keras.layers.add([x, layer_input])
  return x



def leray_projection(fields, eps=1e-8):
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))
  u_rft = fields_rft[..., 0]
  v_rft = fields_rft[..., 1]

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)

  # (1) compute divergence 
  ikxu = 1j * kx_mesh * u_rft[..., jnp.newaxis]
  ikyv = 1j * ky_mesh * v_rft[..., jnp.newaxis]
  div_u_rft = ikxu + ikyv

  # (2) solve Poisson problem 
  phi_rft = - div_u_rft / (eps + kx_mesh ** 2 + ky_mesh ** 2)

  # (3) take grad into channels
  u_correct_ft = -jnp.concatenate([1j * kx_mesh * phi_rft,
                                   1j * ky_mesh * phi_rft], axis=-1)
  u_correction = jnp.fft.irfftn(u_correct_ft, axes=(1,2))
  return fields + u_correction

# objective is make div-func pre-compiled to avoid rebuild every batch
def div_free_2D_layer(u_in):
  """" Custom layer to project onto divergence-free solution via Leray:
          u_out = u_in - grad( nab^{-1} div(u_in) )
       Expected shape is (None, Nx, Ny, 2) """
  if len(u_in.shape) != 4:
    raise ValueError("Expected 4D input, input has shape", u_in.shape)
  if u_in.shape[-1] != 2:
    raise ValueError("Expected 2 channels (2D vel field) but input has ",
                     u_in.shape[-1],
                     "channels.")
  # output shape does not include batch dim -- check
  u_projected = Lambda(leray_projection, 
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected

def exponential_filter(fields):
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)
  
  # following Dresdner et al filter exp(- alpha | k / k_max | ^ 2p | ); p = 32, alpha = 6
  k_all = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
  k_max = jnp.max(k_all)
  filter_exp = jnp.exp( -6 * (k_all / k_max) ** 64)

  # filter field and invert
  filtered_field = fields_rft * filter_exp
  
  return jnp.fft.irfftn(filtered_field, axes=(1,2))

def circular_filter(fields):
  """ Based on JAX-CFD spectral code base; apply 2/3 de-aliasing to output field
      [smooth version; TODO read refs] """
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)
  
  k_all = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
  k_max = jnp.max(k_all)

  # following based on JAX-CFD
  cphi = 0.65 * k_max
  filterfac = 23.6
  filter_ = jnp.exp(-filterfac * (k_all - cphi) ** 4.)
  filter_ = jnp.where(k_all <= cphi, jnp.ones_like(filter_), filter_)
  
  filtered_field = fields_rft * filter_
  return jnp.fft.irfftn(filtered_field, axes=(1,2))

# exp filter layer
def exp_filter_layer(u_in):
  """" Custom layer to apply exponential filter """
  u_projected = Lambda(exponential_filter, 
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected

# circ filter layer
def circ_filter_layer(u_in):
  """ Custom layer for de-aliasing filter """
  u_projected = Lambda(circular_filter,
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected


##################################LZ's Newt######################################
#################################################################################

def newt_single_frame_fpc(
    Nx, Ny, N_filters,
    N_layer=2,               
    N_levels=3,               
    kernel=(3,3),
    input_channels=1,
    output_channels=1,
    filter_factor=2,
):
    """
    Periodic ResNet + Encoder-Decoder:
      (Nx,Ny,Cin) -> (Nx,Ny,Cin+Cout)
    """
    
    input_vort = Input(shape=(Nx, Ny, input_channels), name="frame_input")

    # ---- Stem ----
    x = periodic_convolution(input_vort, N_filters, kernel=kernel,
        n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, activation="linear" )

    # ---- Encoder ----
    skips = []
    filter_schedule = [int(round(N_filters * (filter_factor ** lev)))
                       for lev in range(N_levels)]

    for lev in range(N_levels):
        filters = filter_schedule[lev]

        for _ in range(N_layer):
            x = residual_block_periodic_conv_LZ(x, filters, kernel=kernel, n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, )

        skips.append(x)

        if lev < N_levels - 1:
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

    filters = filter_schedule[-1]
    for _ in range(max(1, N_layer)):
        x = residual_block_periodic_conv_LZ(x, filters, kernel=kernel, n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, )

    # ---- Decoder ----
    for lev in reversed(range(N_levels - 1)):
        filters = filter_schedule[lev]

        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

        x = periodic_convolution(x, filters, kernel=kernel,
            n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, activation="gelu" )

        x = Concatenate(axis=-1)([x, skips[lev]])

        for _ in range(N_layer):
            x = residual_block_periodic_conv_LZ(x, filters, kernel=kernel,
                n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, )


    output = periodic_convolution(x, output_channels, kernel=kernel,
        n_pad_rows=kernel[0]-1, n_pad_cols=kernel[1]-1, activation="linear",)


    return keras.Model(input_vort, output, name="newt_single_frame")  



def newt_VC_traj_fpc(Nx, Ny, Nt, N_filters, N_layer=2, N_levels=3, kernel=(3,3), input_channels=1, output_channels=3, filter_factor=2):
  """ Build a model to perform polymer stress prediction. Apply lattent variables """

  input_vort = Input(shape=(Nt, Nx, Ny, input_channels), name="vort_traj_input")


  frame_model = newt_single_frame_fpc(Nx, Ny, N_filters=N_filters,N_layer=N_layer,N_levels=N_levels, kernel=kernel,
                                  input_channels=input_channels,output_channels=output_channels, filter_factor=filter_factor,)

  output_traj = TimeDistributed(frame_model, name="per_t_cnn")(input_vort)

  return Model(input_vort, output_traj, name="oldb_VC_traj_encdec")    



