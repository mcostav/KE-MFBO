# import libraries
import numpy as np
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import differential_evolution
import json
import os
import datetime

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as sp
from jax.scipy.optimize import minimize as minJAX
from jax.scipy.special import ndtr
import random
import utils_NLL as nll

###########################
# ---Gaussian Processes---#
###########################

############################################################
# --- Minimizing the NLL (hyperparameter optimization) --- #
############################################################   

# --- Differential evolution --- #
def solveDifferentialEvol(bounds_np, X, Y):
    arguments = (X, Y)
    res = differential_evolution(nll.negative_loglikelihood, bounds=bounds_np, args=arguments,
                                         maxiter=10, popsize=15)
    xbest = res.x
    xbest = jnp.array(xbest)
    return xbest         #JAX arrays

# --- Penalty method for NLL --- #
def solveNLL(hyper, X, Y, bounds):
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    lowerDev = jnp.sum(jnp.square(jnp.maximum(lower - hyper, 0)))
    upperDev = jnp.sum(jnp.square(jnp.maximum(hyper - upper, 0)))
    return nll.negative_loglikelihood_jnp(hyper, X, Y) + 100*lowerDev + 50*upperDev    #JAX arrays

# --- Minimize NLL for best hyperparameters --- #
def determine_hyperparameters(X, Y, noise=None):
        '''
        Notice we construct one GP for each y output dim (nydim)
        '''
        #Normalize
        meanX, stdX     = np.mean(np.array(X), axis=0), np.std(np.array(X), axis=0)
        meanY, stdY     = np.mean(np.array(Y), axis=0), np.std(np.array(Y), axis=0)
        X, Y = (X - meanX)/stdX, (Y - meanY)/stdY
        X_norm_jax, Y_norm_jax = jnp.array(X), jnp.array(Y)
        n_point, nx_dim, ny_dim = X.shape[0], X.shape[1], Y.shape[1]

        # No need to normalize this bounds for hyper
        lb               = np.array([-4.]*(nx_dim+1) + [-10.])  # lb on parameters (inside exp.)
        ub               = np.array([4.]*(nx_dim+1) + [ -1.])   # ub on parameters (inside exp.)
        bounds_jnp = jnp.hstack((lb.reshape(nx_dim+2,1),ub.reshape(nx_dim+2,1)))
        bounds_np = list(zip(lb, ub))  # Convert to list of tuples
        
        hypopt   = jnp.zeros((nx_dim+2, ny_dim))  # hyperparams w's + sf2+ sn2 (one for each GP = output y)
        invKopt = []
        
        for i in range(ny_dim):
            # --- multistart loop --- #
            xbest = solveDifferentialEvol(bounds_np, X, Y[:, i])
            res = minJAX(solveNLL, x0=xbest, args =(X_norm_jax, Y_norm_jax[:, i], bounds_jnp),
                           method='BFGS')
            hypopt = hypopt.at[:, i].set(res.x)
            ellopt      = jnp.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = jnp.exp(2.*hypopt[nx_dim,i])
            if noise is not None:
                sn2opt = noise
            else:
                sn2opt      = jnp.exp(2.*hypopt[nx_dim+1,i]) + 1e-8
            # --- constructing optimal K --- #
            Kopt = nll.Cov_mat_jax(X_norm_jax, ellopt, sf2opt) + sn2opt*jnp.eye(n_point)
 
            # --- append inverted K for all nydim --- #
            invKopt.append(jax.numpy.linalg.solve(Kopt, jnp.eye(n_point)))
        
        return hypopt, invKopt      # JAX arrays

########################
# --- GP inference --- #
########################

def GP_inference_np(X, Y, x, hypopt, invKsample):
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
            k       = nll.calc_cov_sample(xnorm, X, ellopt, sf2opt, nx_dim)
            mean[i] = np.squeeze(np.matmul(np.matmul(k.T,invK),Y[:,i]))
            var[i]  = np.squeeze(max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k))) # numerical error
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        return np.squeeze(mean_sample), np.squeeze(var_sample)      #Non-JAX arrays


def GP_inference_jnp(X, Y, x, hypopt, invKsample):
    #Both mean and standard deviation of the non-normalized sets
    meanX, stdX     = jnp.mean(jnp.array(X), axis=0), jnp.std(jnp.array(X), axis=0)
    meanY, stdY     = jnp.mean(jnp.array(Y), axis=0), jnp.std(jnp.array(Y), axis=0)
    #Normalize
    Xsample, Ysample = (X - meanX)/stdX, (Y - meanY)/stdY
    nx_dim, ny_dim = X.shape[1], Y.shape[1]
    
    xnorm = (x - meanX)/stdX
    mean  = jnp.zeros(ny_dim)
    var   = jnp.zeros(ny_dim)
    # --- Loop over each output (GP) --- #
    def compute_mean_var(i):
        invK = invKsample[i]
        hyper = hypopt[:, i]
        ellopt, sf2opt = jnp.exp(2 * hyper[:nx_dim]), jnp.exp(2 * hyper[nx_dim])
        
        # determine covariance of each output
        k = nll.calc_cov_sample_jnp(xnorm, Xsample, ellopt, sf2opt, nx_dim)
        jax.debug.print("k: {}", k)
        m_i = jnp.matmul(jnp.matmul(k.T, invK), Ysample[:, i])
        varCalc = sf2opt - jnp.matmul(jnp.matmul(k.T, invK), k)
        jax.debug.print("variance computed: {}", varCalc)
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

#################################
# --- Bayesian Optimization --- #
#################################

# --- Generate dataset --- #
def compute_data(obj_f, data, noise=None):
        '''
        --- Compute Xtrain and ytrain data, generating x data in the bounds of the region ---
        '''
        # internal variable calls
        ndata = data[2]
        ndim  = data[1].shape[0]
        x_max, x_min = data[1][:,1], data[1][:,0]

        # computing data
        fx     = np.zeros(ndata)
        xsmpl  = sobol_seq.i4_sobol_generate(ndim, ndata) # xsmpl.shape = (ndata,ndim)
        
        Xtrain = np.zeros((ndata, ndim))
        # computing Xtrain
        for i in range(ndata):
            xdat        = x_min + xsmpl[i,:]*(x_max-x_min)
            Xtrain[i,:] = xdat

        for i in range(ndata):
            if noise is not None:
                fx[i] = obj_f(Xtrain[i,:], noise)
            else:
                fx[i] = obj_f(Xtrain[i,:])

        # not meant for multi-output
        ytrain = fx.reshape(ndata,1)
           
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)
        
        return Xtrain.reshape(ndata,ndim, order='F'), ytrain.reshape(ndata,1)       #Non-JAX arrays


def data_handling(obj_f, data, noise=None):
        '''
        --- Use training data supplied or generate it through compute_data ---
        '''
        if data[0]=='int':
            print('- No preliminar data supplied, computing data by sobol sequence')
            if noise is not None:
                Xtrain, ytrain = compute_data(obj_f, data, noise)
            else:
                Xtrain, ytrain = compute_data(obj_f, data)
            return Xtrain, ytrain       #Non-JAX arrays

        elif data[0]=='data0':
            print('- Training data has been suplied')
            Xtrain = data[1][0]
            ytrain = data[1][1]
            return Xtrain, ytrain       #Non-JAX arrays

        else:
            print('- error, data argument ',data,' is of wrong type; can be int or ')
            return None

#######################################
# --- Acquisition functions (JAX) --- #
#######################################

    #####################
    # --- UCB (JAX) --- #
    #####################

def UCB_obj_f_jax(x, X, Y, hypopt, invKsample):
    '''
    Define exploration - exploitation strategy
    '''
    jax.debug.print("x tested: {}", x)
    obj_f = GP_inference_jnp(X, Y, x, hypopt, invKsample)
    res = obj_f[0] - 2 * jnp.sqrt(obj_f[1])
    jax.debug.print("mean: {}", obj_f[0])
    jax.debug.print("variance: {}", obj_f[1])
    return res      #JAX arrays

def solveUCB_jax(x, X, Y, lb, ub, hypopt, invKsample):
    lowerDev = jnp.sum(jnp.square(jnp.maximum(lb - x, 0)))
    upperDev = jnp.sum(jnp.square(jnp.maximum(x - ub, 0)))
    return UCB_obj_f_jax(x, X, Y, hypopt, invKsample) + 100*lowerDev + 75*upperDev     #JAX arrays

def solveUCBObjective(X, Y, x_init, lb, ub, noise=None):
    x_init_JAX = jnp.array(x_init)
    lower, upper = jnp.array(lb), jnp.array(ub)
    
    if noise is not None:
        hypopt, invKsample   = determine_hyperparameters(X, Y, noise)
    else:
        hypopt, invKsample   = determine_hyperparameters(X, Y)
    
    # ALL jax arguments except X and Y
    arguments = (X, Y, lower, upper, hypopt, invKsample)
    res = minJAX(solveUCB_jax, x_init_JAX, args=arguments, method='BFGS')
    
    # refine initial point
    resultVar = np.array(res.x)
    resultFun = np.array(res.fun)
    print(f'Print result boolean:', np.array(res.success))
    return res, resultVar, resultFun        #JAX arrays

    ####################
    # --- EI (JAX) --- #
    ####################

def normal_pdf(x):
    return jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi)      #JAX arrays
   
def EI_obj_f_jax(x, X, Y, hypopt, invKsample):
    '''
    define exploration - explotation strategy
    (mu(x)-f(x)-xi)*Phi(z)+sigma(x)phi(z)
    Z = (mu(x)-f(x)-xi)/sigma(x)
    '''
    # function definitions
    xi     = 0.1
    obj_f  = GP_inference_jnp(X, Y, x, hypopt, invKsample)   # GP evaluation
    Y = jnp.array(Y)
        
    sigma = jnp.sqrt(jnp.array(obj_f[1]))  # Standard deviation is the square root of the variance
    f_plus = jnp.min(Y) # best function value so far
    temp   = f_plus - obj_f[0] + xi
    Z_x    = temp/sigma          # standardized value for normal
    EI_f = temp * ndtr(Z_x) + sigma * normal_pdf(Z_x)     #More JAX compatible
    #EI_f will be a single column array of nydim rows
    
    return -EI_f        #JAX arrays

def solveEI_jax(x, X, Y, lb, ub, hypopt, invKsample):
    lowerDev = jnp.sum(jnp.square(jnp.maximum(lb - x, 0)))
    upperDev = jnp.sum(jnp.square(jnp.maximum(x - ub, 0)))
    return EI_obj_f_jax(x, X, Y, hypopt, invKsample) + 100*lowerDev + 75*upperDev      #JAX arrays

def solveEIObjective(X, Y, x_init, lb, ub, noise=None):
    x_init_JAX = jnp.array(x_init)
    lower, upper = jnp.array(lb), jnp.array(ub)
    
    if noise is not None:
        hypopt, invKsample   = determine_hyperparameters(X, Y, noise)
    else:
        hypopt, invKsample   = determine_hyperparameters(X, Y)
    
    # ALL jax arguments except X and Y
    arguments = (X, Y, lower, upper, hypopt, invKsample)
    res = minJAX(solveEI_jax, x_init_JAX, args=arguments, method='BFGS')
    
    # refine initial point
    resultVar = np.array(res.x)
    resultFun = np.array(res.fun)
    print(f'Print result boolean:', np.array(res.success))
    return res, resultVar, resultFun        #JAX arrays


# --- Creating storing data arrays --- #  

def create_data_arrays(iter_opt, ndim, bounds, ntest, store_data):
    Xtest_l = np.zeros((ntest,ndim))
    if ndim == 1:
        Xtest_l = np.linspace(bounds[0][0], bounds[0][1], num=ntest).reshape(ntest, ndim, order='F')
    else:
        Xtest_l = np.linspace(bounds[:, 0], bounds[:, 1], num=ntest).reshape(ntest, ndim, order='F')
    if store_data == True:
        ymean_l = np.zeros((iter_opt + 1,ntest))
        ystd_l  = np.zeros((iter_opt + 1,ntest))
    else: # Just save last iteration
        ymean_l = np.zeros((1, ntest))
        ystd_l  = np.zeros((1, ntest))
    return Xtest_l, ymean_l, ystd_l     #Non-JAX arrays

# --- Storing data --- #
    
def add_data(Xtest_l, ymean_l, ystd_l, i_opt, X, Y, noise=None): 
    # internal variable calls
    n_test = Xtest_l.shape[0]
    
    if noise is not None:
        hypopt_jax, invKsample_jax   = determine_hyperparameters(X, Y, noise)
    else:
        hypopt_jax, invKsample_jax   = determine_hyperparameters(X, Y)
    
    hypopt, invKsample = np.array(hypopt_jax), np.array(invKsample_jax)
    print('at iteration:', i_opt, 'hyper-parameters values:', hypopt)
    
    for ii in range(n_test):
        m_ii, std_ii      = GP_inference_np(X, Y, Xtest_l[ii,:], hypopt, invKsample)
        ymean_l[i_opt,ii] = m_ii  #updated m and std in the row of that iteration, across the columns
        ystd_l[i_opt,ii]  = np.sqrt(std_ii) #for different combos of different x values (for all dimensions)
    return ymean_l, ystd_l      #Non-JAX arrays

##################################
# --- Optimization algorithm --- #
##################################

def optimization_routine(obj_f, data, iter_opt, bounds, multi_opt, EI_bool, ntest, store_data, noise, seed):
        '''
        data = ['int', bound_list=[[0,10],[-5,5],[3,8]], samples_number] <=> d = ['int', np.array([[-12, 8]]), 3]
        data = ['data0', data=[Xtrain,ytrain]]
        '''
        print('- Note: GP_optimizer works for unconstrained single output GP optimization')
        
        Xtrain, ytrain = data_handling(obj_f, data, noise)
        print(f'Print Xtrain', Xtrain)
        print(f'Print ytrain', ytrain)
        ndim =  Xtrain.shape[1]
        #n_point, ny_dim = Xtrain.shape[0], ytrain.shape[1]
        
        # --- normalize bounds --- #
        lb, ub           = bounds[:,0], bounds[:,1] #1D arrays
        
        # --- storing data --- #
        Xtest_l, ymean_l, ystd_l = create_data_arrays(iter_opt, ndim, bounds, ntest, store_data)
        
        work = True #switch in case it does not work
        # optimization -- iterations
        for i_opt in range(iter_opt):
            print('optimization interation ',i_opt)
            np.random.seed(seed)
            scalarRdn = random.random()
            if scalarRdn <= 0.5:
                multi_startvec   = np.random.rand(multi_opt, ndim) #shape = multi_opt, ndim
            else:
                multi_startvec   = sobol_seq.i4_sobol_generate(ndim, multi_opt) #shape = multi_opt, ndim
            
            if store_data == True:
                ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, i_opt, Xtrain, ytrain, noise)
            
            # optimization -- multistart
            def runMultistart(multi_opt, ndim, i_opt, lb, ub, multi_startvec, EI_bool, Xtrain, ytrain):
                localsol = np.zeros((multi_opt, ndim))  # values for multistart
                localval = np.zeros((multi_opt))    # variables for multistart
                for j in range(multi_opt):
                    print('iter ',i_opt,'  multistart ',j)
                    x_init    = lb + (ub-lb)*multi_startvec[j,:]
                
                    # GP optimization JAX (ALL arguments in JAX)
                    if EI_bool == True:
                        res, resultVar, resultFun = solveEIObjective(Xtrain, ytrain, x_init, lb, ub, noise)
                    else:
                        res, resultVar, resultFun = solveUCBObjective(Xtrain, ytrain, x_init, lb, ub, noise)
                
                    localsol[j] = resultVar
                    if res.success == True:
                        localval[j] = resultFun
                    else:
                        localval[j] = np.inf
                return localsol, localval
            
            localsol, localval = runMultistart(multi_opt, ndim, i_opt, lb, ub, multi_startvec, EI_bool, Xtrain, ytrain)       
            if np.min(localval) == np.inf:
                print('warning, no feasible solution found')
                print('Do not update Xtrain and ytrain')
                print('Try other acquisition functions:')
                
                # Flip the boolean value
                EI_bool = not EI_bool
                localsol, localval = runMultistart(multi_opt, ndim, i_opt, lb, ub, multi_startvec, EI_bool, Xtrain, ytrain)
                
                #Still left to try both in SF and MF a solution to change kernel
                if np.min(localval) == np.inf: #means change of acquisition function does not work
                    print('Reshaping the storage arrays to the former iteration')
                    work = False
                    # This iteration found nothing feasible,
                    # so just reshape to keep size i_opt,
                    # which means from 0 to i_opt-1
                    if store_data == True:
                        ymean_l, ystd_l = ymean_l[:i_opt,:], ystd_l[:i_opt,:]
                    else: #compute the result from last iteration
                        row = 0
                        ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, row, Xtrain, ytrain, noise)
                        ymean_l, ystd_l = ymean_l.flatten(), ystd_l.flatten()
                    break
                else:
                    minindex    = np.argmin(localval) # choosing best solution
                    xnew        = localsol[minindex]  # selecting best solution

                    xnew   = np.array([xnew]).flatten()
                    ynew   = obj_f(xnew, noise)
                    # adding new point to sample
                    Xtrain  = np.vstack((Xtrain,xnew))
                    ytrain  = np.vstack((ytrain,ynew))
            else:
                minindex    = np.argmin(localval) # choosing best solution
                xnew        = localsol[minindex]  # selecting best solution

                xnew   = np.array([xnew]).flatten()
                ynew   = obj_f(xnew, noise)
                # adding new point to sample
                Xtrain  = np.vstack((Xtrain,xnew))
                ytrain  = np.vstack((ytrain,ynew))
        
        # --- storing final data (on i_opt+1) --- #
        # i_opt + 1 = iter_tot
        if work == True and store_data == True:
            ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, i_opt+1, Xtrain, ytrain, noise)
        elif work == True and store_data == False:
            row = 0
            ymean_l, ystd_l = add_data(Xtest_l, ymean_l, ystd_l, row, Xtrain, ytrain, noise)
            ymean_l, ystd_l = ymean_l.flatten(), ystd_l.flatten()
        return Xtrain, ytrain, Xtest_l, ymean_l, ystd_l, i_opt

#################################
# --- Bayesian Optimization --- #
#################################

def BO_np_scipy(f, bounds, iter_tot, multi_opt, EI_bool, store_data, noise, seed):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''
    n_rs   = int(min(100,max(iter_tot*.05,5)))  # iterations to find good starting point
    d_     = ['int', bounds, n_rs]  #no data provided
    ntest = 200 # number of samples for trajectory

    # i_opt + 1 = iter_tot
    # evaluate first point
    X_opt, y_opt, Xtest_l, ymean_l, ystd_l, i_opt  = optimization_routine(f, d_, iter_tot, bounds, multi_opt, EI_bool, ntest, store_data, noise, seed)
    X_opt, y_opt = np.array(X_opt), np.array(y_opt)
    i_best       = np.argmin(y_opt)
    x_best       = X_opt[i_best, :]
    y_best       = y_opt[i_best, :]
    
    return x_best, y_best, Xtest_l, ymean_l, ystd_l, X_opt, y_opt, n_rs, i_opt