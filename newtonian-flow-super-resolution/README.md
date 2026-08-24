## Super-resolution of turbulence with dynamics through TraCTra

A set of routines to train neural networks to perform super-resolution, without necessarily requiring high-resolution data.
The approach is to instead train the neural network with a loss function that is similar to 4DVar data-assimilation, requiring that the 
coarse-grained forward trajectory of the model matches reference data. 

Implementation wraps around the spectral version of JAX-CFD (https://github.com/google/jax-cfd). Neural networks are written in Keras using the JAX backend.  
