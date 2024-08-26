import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as sp
from jax.scipy.optimize import minimize as minJAX

# --- JAX Negative log-likelihood --- #

def Cov_mat_jax(X_norm, W, sf2):
    '''
    Calculates the covariance matrix of a dataset Xnorm
    --- RBF ---
    '''
    W_sqrt = jnp.sqrt(W)
    scaled_X = X_norm / W_sqrt
    dist_squared = jnp.sum((scaled_X[:, None, :] - scaled_X[None, :, :]) ** 2, axis=-1)
    cov_matrix = sf2 * jnp.exp(-0.5 * dist_squared)
    return cov_matrix       #JAX arrays

def calc_cov_sample_jnp(xnorm, Xnorm, ell, sf2, nx_dim):
    '''
    Calculates the covariance of a single sample xnorm against the dataset Xnorm
    '''
    #xnorm is a single 1D array (nx_dim,) and Xnorm is an array (n_point, nx_dim)
    dist = jnp.sum(((Xnorm - xnorm.reshape(1, nx_dim)) / ell)**2, axis=1).reshape(-1, 1)
    cov_matrix = sf2 * jnp.exp(-0.5 * dist)
    return cov_matrix       #JAX arrays

def Cov_matern_jax(X_norm, W, sf2):
    '''
    Calculates the covariance matrix of a dataset Xnorm
    --- RBF ---
    '''
    W_sqrt = jnp.sqrt(W)
    scaled_X = X_norm / W_sqrt
    dist_squared = jnp.sum((scaled_X[:, None, :] - scaled_X[None, :, :]) ** 2, axis=-1)
    sqrt_5 = jnp.sqrt(5)
    term1 = 1 + sqrt_5 * jnp.sqrt(dist_squared) + (5 * dist_squared) / 3
    term2 = jnp.exp(-sqrt_5 * jnp.sqrt(dist_squared))
    cov = sf2 * term1 * term2

    return cov #JAX arrays

def calc_cov_sample_matern_jnp(xnorm, Xnorm, ell, sf2, nx_dim):
    '''
    Calculates the covariance of a single sample xnorm against the dataset Xnorm
    '''
    #xnorm is a single 1D array (nx_dim,) and Xnorm is an array (n_point, nx_dim)
    dist_squared = jnp.sum(((Xnorm - xnorm.reshape(1, nx_dim)) / ell)**2, axis=1).reshape(-1, 1)
    sqrt_5 = jnp.sqrt(5)
    term1 = 1 + sqrt_5 * jnp.sqrt(dist_squared) + (5 * dist_squared) / 3
    term2 = jnp.exp(-sqrt_5 * jnp.sqrt(dist_squared))
    cov = sf2 * term1 * term2

    return cov #JAX arrays

def negative_loglikelihood_jnp(hyper, X, Y, kernel_f):
        n_point, nx_dim   = X.shape[0], X.shape[1]
        
        W               = jnp.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = jnp.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = jnp.exp(2*hyper[nx_dim+1])  # variance of noise

        if kernel_f == True:
            K       = Cov_mat_jax(X, W, sf2)  # (nxn) covariance matrix (noise free)
        else:
            K       = Cov_matern_jax(X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*jnp.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        
        # Cholesky decomposition using JAX
        L = sp.linalg.cho_factor(K, lower=True)[0]
        logdetK = 2 * jnp.sum(jnp.log(jnp.diag(L)))  # calculate log of determinant of K
        Y = jnp.array(Y)
        invLY = sp.linalg.cho_solve((L, True), Y)  # solve L * L.T * alpha = Y
        NLL = jnp.dot(Y.T, invLY) + logdetK  # construct the NLL

        return NLL       #JAX scalar
    
# --- Non-JAX Negative log-likelihood --- #

def Cov_mat(X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- RBF ---
        Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        '''
        dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2 
        cov_matrix = sf2*np.exp(-0.5*dist)
        return cov_matrix       #Non-JAX arrays
        # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        
def calc_cov_sample(xnorm, Xnorm, ell, sf2, nx_dim):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        '''
        #xnorm is a single 1D array (nx_dim,) and Xnorm is an array (n_point, nx_dim)
        dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-0.5*dist)

        return cov_matrix            #Non-JAX arrays
    
def Cov_matern(X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- Matern 5/2 ---
        Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        '''
        dist_squared = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
        sqrt_5 = np.sqrt(5)
        term1 = 1 + sqrt_5 * np.sqrt(dist_squared) + (5 * dist_squared) / 3
        term2 = np.exp(-sqrt_5 * np.sqrt(dist_squared))
        cov_matrix = sf2 * term1 * term2
        return cov_matrix       #Non-JAX arrays
        # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])

        
def calc_cov_matern_sample(xnorm, Xnorm, ell, sf2, nx_dim):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        '''
        #xnorm is a single 1D array (nx_dim,) and Xnorm is an array (n_point, nx_dim)
        dist_squared = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        sqrt_5 = jnp.sqrt(5)
        term1 = 1 + sqrt_5 * np.sqrt(dist_squared) + (5 * dist_squared) / 3
        term2 = np.exp(-sqrt_5 * np.sqrt(dist_squared))
        cov_matrix = sf2 * term1 * term2

        return cov_matrix            #Non-JAX arrays

def negative_loglikelihood(hyper, X, Y, kernel_f):
        #pre-data
        n_point, nx_dim   = X.shape[0], X.shape[1]
        
        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise
        
        if kernel_f == True:
            K       = Cov_mat(X, W, sf2)  # (nxn) covariance matrix (noise free)
        else:
            K       = Cov_matern(X, W, sf2)  # (nxn) covariance matrix (noise free)

        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL      #Non-JAX scalar

############################################################
# --- Minimizing the NLL (hyperparameter optimization) --- #
############################################################   

# --- Differential evolution --- #
def solveDifferentialEvol(bounds_np, X, Y, kernel_f):
    arguments = (X, Y, kernel_f)
    res = differential_evolution(negative_loglikelihood, bounds=bounds_np, args=arguments,
                                         maxiter=10, popsize=15)
    xbest = res.x
    xbest = jnp.array(xbest)
    return xbest         #JAX arrays

# --- Penalty method for NLL --- #
def solveNLL(hyper, X, Y, bounds, kernel_f):
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    lowerDev = jnp.sum(jnp.square(jnp.maximum(lower - hyper, 0)))
    upperDev = jnp.sum(jnp.square(jnp.maximum(hyper - upper, 0)))
    return negative_loglikelihood_jnp(hyper, X, Y, kernel_f) + 100*lowerDev**2 + 50*upperDev**2    #JAX arrays

# --- Minimize NLL for best hyperparameters --- #
def determine_hyperparameters(X, Y, kernel_f, noise = None):
        '''
        Notice we construct one GP for each y output dim (nydim)
        '''
        #Normalize
        meanX, stdX     = np.mean(np.array(X), axis=0), np.std(np.array(X), axis=0)
        meanY, stdY     = np.mean(np.array(Y), axis=0), np.std(np.array(Y), axis=0)
        X, Y = (X - meanX)/stdX, (Y - meanY)/stdY
        X_norm_jax, Y_norm_jax = jnp.array(X), jnp.array(Y)
        n_point, nx_dim, ny_dim = X.shape[0], X.shape[1], Y.shape[1]
        
        
        lb               = np.array([-4.]*(nx_dim+1) + [-10.])  # lb on parameters (inside exp.)
        ub               = np.array([4.]*(nx_dim+1) + [ -1.])   # ub on parameters (inside exp.)
        bounds_jnp = jnp.hstack((lb.reshape(nx_dim+2,1),ub.reshape(nx_dim+2,1)))
        bounds_np = list(zip(lb, ub))  # Convert to list of tuples
        
        hypopt   = jnp.zeros((nx_dim+2, ny_dim))  # hyperparams w's + sf2+ sn2 (one for each GP = output y)
        invKopt = []
        
        for i in range(ny_dim):
            # --- multistart loop --- #
            xbest = solveDifferentialEvol(bounds_np, X, Y[:, i], kernel_f)
            res = minJAX(solveNLL, x0=xbest, args =(X_norm_jax, Y_norm_jax[:, i], bounds_jnp, kernel_f),
                           method='BFGS')
            jax.debug.print("success of hyper-parameter optimisation: {}", res.success)
            
            hypopt = hypopt.at[:, i].set(res.x)
            ellopt      = jnp.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = jnp.exp(2.*hypopt[nx_dim,i])
            if noise is not None:
                sn2opt = noise
            else:
                sn2opt      = jnp.exp(2.*hypopt[nx_dim+1,i]) + 1e-8
            # --- constructing optimal K --- #
            if kernel_f == True: #RBF
                Kopt = Cov_mat_jax(X_norm_jax, ellopt, sf2opt) + sn2opt*jnp.eye(n_point)
            else: #Matern 5/2
                Kopt = Cov_matern_jax(X_norm_jax, ellopt, sf2opt) + sn2opt*jnp.eye(n_point)
 
            # --- append inverted K for all nydim --- #
            invKopt.append(jax.numpy.linalg.solve(Kopt, jnp.eye(n_point)))
        
        return hypopt, invKopt, ellopt, sf2opt      # JAX arrays