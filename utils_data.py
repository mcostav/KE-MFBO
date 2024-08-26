# import libraries
import numpy as np
import sobol_seq
import utils_NLL as nll
import utils_gp as gp

# --- Generate dataset --- #
def compute_data(obj_f, cost_f, data, noise=None):
        '''
        --- Compute Xtrain and ytrain data, generating x data in the bounds of the region ---
        '''
        # internal variable calls
        ndata = data[3]
        ndim  = data[1].shape[0]
        ndim_fidel = data[2].shape[0]
        x_max, x_min = data[1][:,1], data[1][:,0]
        z_max, z_min = data[2][:,1], data[2][:,0]

        # computing data
        fx     = np.zeros(ndata)
        cx     = np.zeros(ndata)
        xsmpl  = sobol_seq.i4_sobol_generate(ndim, ndata) # xsmpl.shape = (ndata,ndim)
        zsmpl  = sobol_seq.i4_sobol_generate(ndim_fidel, ndata) # xsmpl.shape = (ndata,ndim)
        
        Xtrain = np.zeros((ndata, ndim))
        ztrain = np.zeros((ndata, ndim_fidel))
        # computing Xtrain
        for i in range(ndata):
            xdat        = x_min + xsmpl[i,:]*(x_max-x_min)
            Xtrain[i,:] = xdat
            zdat        = z_min + zsmpl[i,:]*(z_max-z_min)
            ztrain[i,:] = zdat

        for i in range(ndata):
            if noise is not None:
                fx[i] = obj_f(Xtrain[i,:], ztrain[i,:], noise)
                cx[i] = cost_f(ztrain[i,:])
            else:
                fx[i] = obj_f(Xtrain[i,:], ztrain[i,:])
                cx[i] = cost_f(ztrain[i,:])

        # not meant for multi-output
        ytrain = fx.reshape(ndata,1)
        ctrain = cx.reshape(ndata,1)
           
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)
        ztrain = np.array(ztrain)
        ctrain = np.array(ctrain)
        
        return Xtrain.reshape(ndata,ndim, order='F'), ytrain.reshape(ndata,1), ztrain.reshape(ndata,ndim_fidel, order='F'), ctrain.reshape(ndata,1)
    #Non-JAX arrays

def data_handling(obj_f, cost_f, data, noise):
        '''
        --- Use training data supplied or generate it through compute_data ---
        '''
        if data[0]=='int':
            print('- No preliminar data supplied, computing data by sobol sequence')
            Xtrain, ytrain, ztrain, ctrain = compute_data(obj_f, cost_f, data, noise)
            return Xtrain, ytrain, ztrain, ctrain       #Non-JAX arrays

        elif data[0]=='data0':
            print('- Training data has been suplied')
            Xtrain = data[1][0]
            ytrain = data[1][1]
            ztrain = data[1][2]
            ctrain = data[1][3]
            return Xtrain, ytrain, ztrain, ctrain       #Non-JAX arrays

        else:
            print('- error, data argument ',data,' is of wrong type; can be int or ')
            return None

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
        cmean_l = np.zeros((iter_opt + 1,ntest))
        cstd_l  = np.zeros((iter_opt + 1,ntest))
    else: # Just save last iteration
        ymean_l = np.zeros((1, ntest))
        ystd_l  = np.zeros((1, ntest))
        cmean_l = np.zeros((1, ntest))
        cstd_l  = np.zeros((1, ntest))
    return Xtest_l, ymean_l, ystd_l, cmean_l, cstd_l     #Non-JAX arrays

# --- Storing data --- #
def add_data(Xtest_l, ymean_l, ystd_l, cmean_l, cstd_l, i_opt, X, Y, C, noise, kernel_f):
    n_test = Xtest_l.shape[0]
    
    hypopt_jax, invKsample_jax, ellopt, sf2opt    = nll.determine_hyperparameters(X, Y, kernel_f, noise)
    hypopt, invKsample = np.array(hypopt_jax), np.array(invKsample_jax)
    print('at iteration:', i_opt, 'hyper-parameters values:', hypopt)
    
    hypoptCost_jax, invKsampleCost_jax, ellopt_Cost, sf2opt_Cost    = nll.determine_hyperparameters(X, C, kernel_f, noise)
    hypoptCost, invKsampleCost = np.array(hypoptCost_jax), np.array(invKsampleCost_jax)
    print('at iteration:', i_opt, 'hyper-parameters cost values:', hypopt)
    
    #X and Xtest should be unified!!
    for ii in range(n_test):
        m_ii, std_ii      = gp.GP_inference_np(X, Y, Xtest_l[ii,:], hypopt, invKsample, kernel_f)
        ymean_l[i_opt,ii] = m_ii  #updated m and std in the row of that iteration, across the columns
        ystd_l[i_opt,ii]  = np.sqrt(std_ii) #for different combos of different x values (for all dimensions)
        c_ii, c_std_ii      = gp.GP_inference_np(X, C, Xtest_l[ii,:], hypoptCost, invKsampleCost, kernel_f)
        cmean_l[i_opt,ii] = c_ii  #updated m and std in the row of that iteration, across the columns
        cstd_l[i_opt,ii]  = np.sqrt(c_std_ii) #for different combos of different x values (for all dimensions)

    return ymean_l, ystd_l, cmean_l, cstd_l      
    #Non-JAX arrays