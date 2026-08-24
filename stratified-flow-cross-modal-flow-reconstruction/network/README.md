## Predicting velocities from density in a stratified turbulent flows through TraCTra

A set of routines to train neural networks to perform cross-modal prediction, without necessarily requiring velocity data.
The approach is to instead train the neural network with a loss function that is similar to 4DVar data-assimilation. 



Implementation wraps around the spectral version of JAX-CFD (https://github.com/google/jax-cfd). Neural networks are written in Keras using the JAX backend.  
