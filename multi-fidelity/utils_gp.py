# import libraries
import numpy as np
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import differential_evolution
import random

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as sp
from jax.scipy.optimize import minimize as minJAX
from jax import grad
import utils_NLL as nll
import utils_data as ut

###########################
# ---Gaussian Processes---#
###########################

def GP_inference_np(X, Y, x, hypopt, invKsample, kernel_f):
        #Normalize
        #Both mean and standard deviation of the non-normalized sets
        meanX, stdX     = np.mean(np.array(X), axis=0), np.std(np.array(X), axis=0)
        meanY, stdY     = np.mean(np.array(Y), axis=0), np.std(np.array(Y), axis=0)
        X, Y = (X - meanX)/stdX, (Y - meanY)/stdY
        
        nx_dim, ny_dim = X.shape[1], Y.shape[1]

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])

            # --- determine covariance of each output --- #
            if kernel_f == True:
                k       = nll.calc_cov_sample(xnorm, X, ellopt, sf2opt, nx_dim)
            else:
                k       = nll.calc_cov_matern_sample(xnorm, X, ellopt, sf2opt, nx_dim)
            mean[i] = np.squeeze(np.matmul(np.matmul(k.T,invK),Y[:,i]))
            var[i]  = np.squeeze(max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k))) # numerical error

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        return np.squeeze(mean_sample), np.squeeze(var_sample)      #Non-JAX arrays


def GP_inference_jnp(X, Y, x, hypopt, invKsample, kernel_f):
    #Both mean and standard deviation of the non-normalized sets
    meanX, stdX     = jnp.mean(jnp.array(X), axis=0), jnp.std(jnp.array(X), axis=0)
    meanY, stdY     = jnp.mean(jnp.array(Y), axis=0), jnp.std(jnp.array(Y), axis=0)
    #jax.debug.print("meanX: {}", meanX)
    #Normalize
    Xsample, Ysample = (X - meanX)/stdX, (Y - meanY)/stdY
    nx_dim, ny_dim = X.shape[1], Y.shape[1]
    
    xnorm = (x - meanX)/stdX
    #jax.debug.print("xnorm: {}", xnorm)
    #jax.debug.print("Xsample: {}", Xsample)
    mean  = jnp.zeros(ny_dim)
    var   = jnp.zeros(ny_dim)
    # --- Loop over each output (GP) --- #
    def compute_mean_var(i):
        invK = invKsample[i]
        hyper = hypopt[:, i]
        ellopt, sf2opt = jnp.exp(2 * hyper[:nx_dim]), jnp.exp(2 * hyper[nx_dim])
        
        # determine covariance of each output
        if kernel_f == True:
            k = nll.calc_cov_sample_jnp(xnorm, Xsample, ellopt, sf2opt, nx_dim)
        else:
            k = nll.calc_cov_sample_matern_jnp(xnorm, Xsample, ellopt, sf2opt, nx_dim)
        #jax.debug.print("covariance between two points: {}", k)
        m_i = jnp.matmul(jnp.matmul(k.T, invK), Ysample[:, i])
        v_i = jnp.maximum(0, sf2opt - jnp.matmul(jnp.matmul(k.T, invK), k))  # numerical error

        # Scalar check so no matrix is used for m_i and v_i
        m_i = jnp.squeeze(m_i)
        v_i = jnp.squeeze(v_i)

        return m_i, v_i
    # --- Loop over each output (GP) --- #
    for i in range(ny_dim):
        mean_i, var_i = compute_mean_var(i)
        mean = mean.at[i].set(mean_i)
        var = var.at[i].set(var_i)
        
    # --- compute un-normalized mean --- #    
    mean_sample = mean*stdY + meanY
    var_sample  = var*stdY**2
        
    return jnp.squeeze(mean_sample), jnp.squeeze(var_sample)    #JAX arrays