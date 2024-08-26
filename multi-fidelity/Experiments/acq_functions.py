# import libraries
import numpy as np

import jax
import optax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as sp
from jax import grad
import utils_NLL as nll
import utils_data as ut
import utils_gp as gp

#######################################
# --- Acquisition functions (JAX) --- #
#######################################

def optimise_acq(X, Y, C, x_init, lb, ub, nx_dim, fid_bounds, noise, kernel_f, final):
    # Set up the bounds for x and initialise parameters (x initial guesses)
    initial_params = jnp.array(x_init)
    bounds_fid = jnp.array(fid_bounds)
    lower, upper = jnp.array(lb), jnp.array(ub)
    
    # Gaussian Process Y (obj_f outputs)
    hypopt, invKsample, ellopt, sf2opt                = nll.determine_hyperparameters(X, Y, kernel_f, noise)
    # Gaussian Process C (ctrain data generated with same Xtrain & ztrain set as the ytrain)
    hypopt_cost, invKsample_cost, ellCost, sf2Cost    = nll.determine_hyperparameters(X, C, kernel_f, noise)

    # Initialize the optimizer (e.g., Adam)
    optimizer = optax.adam(learning_rate=0.001, eps_root=1e-6)

    # Initialize optimizer state
    opt_state = optimizer.init(initial_params)

    def acqSavage_jax(x): 
        #Cost GP
        cost_f = gp.GP_inference_jnp(X, C, x, hypopt_cost, invKsample_cost, kernel_f)
        cost_offset = jnp.squeeze(min(C)) * 3
        
        #Objective GP (at the highest fidelity z)
        x_copy = jnp.copy(x)
        x_conc = jnp.concatenate([x_copy[:nx_dim], bounds_fid[:,1]])
        
        ndim = jnp.shape(x_conc)[0]
        obj_f = gp.GP_inference_jnp(X, Y, x_conc, hypopt, invKsample, kernel_f)
        
        # num = obj_f[0] + 2.5 * jnp.sqrt(obj_f[1])
        num = obj_f[0] - jnp.sqrt(1 * obj_f[1] + 1e-6)
        #jax.debug.print("num: {}", num)
        if kernel_f == True:
            kernel = jnp.squeeze(nll.calc_cov_sample_jnp(x_conc, x, ellopt, sf2opt, ndim)/(sf2opt))
        else:
            kernel = jnp.squeeze(nll.calc_cov_sample_matern_jnp(x_conc, x, ellopt, sf2opt, ndim)/(sf2opt))
        
        denom = 0.5 * (cost_f[0] - cost_offset) * jnp.sqrt(1 - kernel**2 + 1e-6)

        res = num/denom
        
        #jax.debug.print("result acquisition function: {}", -res)
        
        return (-res)

    def greedy_jax(x):
        x_obj = jnp.concatenate([x, bounds_fid[:,1]])
        obj_f = gp.GP_inference_jnp(X, Y, x_obj, hypopt, invKsample, kernel_f)
        
        return -abs(obj_f[0])
    
    # Create a function that returns both value and gradient
    if final == True:
        value_and_grad_fn = jax.value_and_grad(greedy_jax)
    else:
        value_and_grad_fn = jax.value_and_grad(acqSavage_jax)
    
    # Define a single optimization step
    def step_fn(carry, _):
        params, opt_state = carry
        
        # Compute the value and gradients in one pass
        value, grads = value_and_grad_fn(params)

        # Apply the optimizer update to the gradients
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Clip the parameters to ensure they stay within the bounds
        params = jnp.clip(params, lower, upper)
        
        # Return the updated carry (note that a, b, c remain unchanged)
        return (params, opt_state), (value)
    
    # Number of iterations
    num_iterations = 50

    # Use jax.lax.scan to perform the optimization
    initial_carry = (initial_params, opt_state)
    (carry, values) = jax.lax.scan(step_fn, initial_carry, None, length=num_iterations)

    # Extract the final parameters (only x is optimized)
    params, opt_state = carry
    values = values

    jax.debug.print("Final optimized parameters for acquisition_fn: {}", params)
    jax.debug.print("Final acquisition function value: {}", values[-1])
    params_np = np.array(params)
    resultFun = np.squeeze(values[-1])
    
    
    return params_np, resultFun
