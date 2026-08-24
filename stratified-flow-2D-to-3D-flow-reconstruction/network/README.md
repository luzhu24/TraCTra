## Predicting the 3D velocities and density fields from a 2D shadowgraph

A set of routines to train neural networks to perform 2D-to-3D prediction, without necessarily requiring 3D data.
The approach is to instead train the neural network with a loss function that is similar to 4DVar data-assimilation. 

Implementation wraps around the spectral version of JAX-CFD (https://github.com/google/jax-cfd). Neural networks are written in Keras using the JAX backend.  
