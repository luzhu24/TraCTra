## Image complation of flow past cylinder through TraCTra

A set of routines to train neural networks to perform image completation of flow past cylinder, without necessarily requiring completed images. The approach is to instead train the neural network with a loss function that is similar to 4DVar data-assimilation. 

Implementation wraps around the spectral version of JAX-CFD (https://github.com/google/jax-cfd). Neural networks are written in Keras using the JAX backend.  
