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
import utils_gp as gp
import acq_functions as acq

#################################
# --- Bayesian Optimization --- #
#################################

def optimization_routine(obj_f, cost_f, data, budget, bounds_og, bounds_fid, multi_opt, noise, seed, kernel_f):
        '''
        data = ['int', bound_list_x=[[0,10],[-5,5],[3,8]], bound_list_z=[[0,10],[-5,5],[3,8]], samples_number] 
        <=> d = ['int', np.array([[-12, 8]]), np.array([[1, 5]]), 3]
        data = ['data0', data=[Xtrain,ytrain,ztrain,ctrain]]
        '''
        print('- Note: GP_optimizer works for unconstrained one output GP optimization')
        
        # data creating
        Xtrain_og, ytrain, ztrain, ctrain = ut.data_handling(obj_f, cost_f, data, noise)
        print(f'Print Xtrain', Xtrain_og)
        print(f'Print ytrain', ytrain)
        print(f'Print ztrain', ztrain)
        print(f'Print ctrain', ctrain)
        nx_dim = Xtrain_og.shape[1]
        ndim_fidel = ztrain.shape[1]
        
        #Unite x and z
        bounds = np.vstack([bounds_og, bounds_fid]) # shape = (ndim, 2)
        Xtrain = np.hstack([Xtrain_og, ztrain]) # shape = (n_rs, ndim)
        
        # --- optimization loop --- #
        lb, ub           = bounds[:,0], bounds[:,1] #1D arrays
        
        # optimization -- iterations
        i_opt = 1
        normal_operation = True
        while normal_operation == True:
            print('optimization interation ',i_opt)
            
            def runMultistart(final):
                if final==True:
                    ndim = nx_dim
                else:
                    ndim = nx_dim + ndim_fidel
                # Robust multi-start (deterministic-stochastic)
                np.random.seed(seed)
                scalarRdn = random.random()
                if scalarRdn >= 0.3:
                    multi_startvec   = np.random.rand(multi_opt, ndim) #shape = multi_opt, ndim
                else:
                    multi_startvec   = sobol_seq.i4_sobol_generate(ndim, multi_opt) #shape = multi_opt, ndim
                localsol = np.zeros((multi_opt, ndim))  # values for multistart
                localval = np.zeros((multi_opt))    # variables for multistart

                for j in range(multi_opt):
                    print('iter ',i_opt,'  multistart ',j)
                    if final == True:
                        lb_g = lb[:nx_dim]
                        ub_g = ub[:nx_dim]
                        x_init    = lb_g + (ub_g-lb_g)*multi_startvec[j,:]
                        lower = lb_g
                        upper = ub_g
                    else:
                        x_init    = lb + (ub-lb)*multi_startvec[j,:]
                        lower = lb
                        upper = ub
                    
                    # GP optimization JAX (ALL arguments in JAX)
                    resultVar, resultFun = acq.optimise_acq(Xtrain, ytrain, ctrain, x_init, lower, upper, nx_dim, bounds_fid, noise, kernel_f, final)
                    
                    localsol[j, :] = resultVar
                    localval[j] = resultFun

                return localsol, localval
                 
            localsol, localval = runMultistart(final=False)
            
            if np.min(localval) == np.nan:
                print("warning, no feasible solution found")
                print('Do not update Xtrain and ytrain')
                print('Exit loop')
                
                break
            
            minindex    = np.argmin(localval) # choosing best solution
            xnew        = localsol[minindex, :]  # selecting best solution

            xnew, znew = xnew[:nx_dim].flatten(), xnew[nx_dim:].flatten()
            # znew[znew >= 0.85] = 1
            # Generate a random scalar between 0 and 1
            scalar_rdn = random.random()

            # If the random scalar is greater than or equal to 0.3, update the array
            if scalar_rdn <= 0.2:
                # Set dimensions of znew if less than or equal to 0.3 to 1
                znew[znew <= 0.25] = 1
            elif scalar_rdn > 0.2:
                znew[znew >= 0.9] = 1
            
            ynew   = obj_f(xnew, znew, noise)
            cnew   = cost_f(znew)

            x_gp = np.hstack([xnew, znew])
            hypopt_cost, invKsample_cost, ellopt, sf2opt   = nll.determine_hyperparameters(Xtrain, ctrain, kernel_f, noise)
            mean_standard, var_standard = gp.GP_inference_np(Xtrain, ctrain, x_gp, hypopt_cost, invKsample_cost, kernel_f)
                
            max_time_standard = mean_standard + 2 * np.sqrt(var_standard)

            # adding new point to sample
            Xtrain_og  = np.vstack([Xtrain_og,xnew])
            ytrain  = np.vstack([ytrain,ynew])
            ztrain  = np.vstack([ztrain,znew])
            ctrain  = np.vstack([ctrain,cnew])
            Xtrain = np.hstack([Xtrain_og, ztrain]) #shape = (n_rs + i_opt, ndim)
                
            i_opt = i_opt + 1
            budget = budget - max_time_standard
            print("budget available", budget)

            #What Savage's algorithm would have done:
            if budget <= cost_f(np.ones(ndim_fidel)):
                normal_operation = False
                localsol, localval = runMultistart(final=True)
                #What Savage's algorithm would have done:

                minindex    = np.argmin(localval) # choosing best solution
                xnew_Tom        = localsol[minindex, :]  # selecting best solution
                znew_Tom = bounds_fid[:,1]  # Maximum fidelity
                ynew_Tom   = obj_f(xnew, znew, noise)

                #My implementation
                # Utilise best values so far
                i_three_best = np.argsort(ytrain)[:3]
                x_three_best = Xtrain_og[i_three_best]
                rows_ones = np.where(np.all(ztrain == 1, axis=1))[0]
                if len(rows_ones) > 0:
                    # Proceed if there are rows with all values == 1
                    row_best = rows_ones[np.argmin(ytrain[rows_ones])]
                    x_init_best = Xtrain_og[row_best]
                    
                    # Stack x_init_best with x_three_best
                    x_init_set = np.vstack([x_three_best.reshape(3, nx_dim), x_init_best])
                else:
                    # If no rows with all values == 1, skip x_init_best
                    x_init_set = x_three_best.reshape(3, nx_dim)

                lb_three_best = lb[:nx_dim]
                ub_three_best = ub[:nx_dim]
                for i in range(x_init_set.shape[0]):
                    resultVar, resultFun = acq.optimise_acq(Xtrain, ytrain, ctrain, x_init_set[i], lb_three_best, ub_three_best, nx_dim, bounds_fid, noise, kernel_f, final=True)
                    
                    localsol = np.vstack([localsol, resultVar])
                    localval = np.concatenate([localval, np.array([resultFun])])
                
                minindex    = np.argmin(localval) # choosing best solution
                xnew        = localsol[minindex, :]  # selecting best solution
                znew = bounds_fid[:,1]  # Maximum fidelity
                ynew   = obj_f(xnew, znew, noise)
                cnew   = cost_f(znew)
                # adding new point to sample
                Xtrain_og  = np.vstack([Xtrain_og,xnew])
                ytrain  = np.vstack([ytrain,ynew])
                ztrain  = np.vstack([ztrain,znew])
                ctrain  = np.vstack([ctrain,cnew])

                # Find the indices of rows where all elements are 1
                rows_with_all_ones = np.where(np.all(ztrain == 1, axis=1))[0]

                # Find the index of the minimum value in ytrain among the selected rows
                i_best_final = rows_with_all_ones[np.argmin(ytrain[rows_with_all_ones])]

                # Extract the corresponding rows from Xtrain_og, ytrain, ztrain, ctrain
                x_best_output = Xtrain_og[i_best_final, :]
                y_best_output = ytrain[i_best_final, :]
                z_best_output = ztrain[i_best_final, :]
                c_best_output = ctrain[i_best_final, :]

        
        return Xtrain_og, ytrain, ztrain, ctrain, xnew_Tom, znew_Tom, ynew_Tom, x_best_output, y_best_output, z_best_output, c_best_output, i_opt
    
#################################
# --- Bayesian Optimization --- #
#################################

def MFBO_np_scipy(f, cost_f, bounds_og, bounds_fid, budget, multi_opt, noise, seed, kernel_f):
    
    n_rs = 5 # number of initial dataset size (generated with sobol seq.)
    d_     = ['int', bounds_og, bounds_fid, n_rs]  #inputs and fidelities

    Xtrain_og, ytrain, ztrain, ctrain, xnew_Tom, znew_Tom, ynew_Tom, x_best_output, y_best_output, z_best_output, c_best_output, i_opt = optimization_routine(f, cost_f, d_, budget, bounds_og, bounds_fid, multi_opt, noise, seed, kernel_f)

    return {
        'x_best': x_best_output,
        'y_best': y_best_output,
        'z_best': z_best_output,
        'c_best': c_best_output,
        'Xtrain': Xtrain_og,
        'ytrain': ytrain,
        'ztrain': ztrain,
        'ctrain': ctrain,
        'xnew_Tom': xnew_Tom, 
        'znew_Tom': znew_Tom, 
        'ynew_Tom': ynew_Tom,
        'ndata': n_rs,
        'iterations': i_opt,
    }