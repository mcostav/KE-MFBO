import bayesian_optimization as bo
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
import json
import pickle
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# Define objective functions

def objective_function_1(x, noise=None):
    # Currin exponential (has global maxima, thus -res)
    fracOne = 1
    fracTwo = 2300*x[0]**3 + 1900*x[0]**2 + 2092*x[0] + 60
    fracThree = 100*x[0]**3 + 500*x[0]**2 + 4*x[0] + 20
    res = fracOne*fracTwo/fracThree
    if noise is not None:
        mean = 0
        return -res + np.random.normal(mean, np.sqrt(noise))
    else:
        return -res

def objective_function_2(x, noise=None):
    # Branin (has global minima, thus res)
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    res = a*(x[1] - b*x[0] + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s
    if noise is not None:
        mean = 0
        return res + np.random.normal(mean, np.sqrt(noise))
    else:
        return res

def cost_function_1(z):
    return 1 + np.sum(z)

def cost_function_2(z):
    return 1 + 5 * np.sum(z)

def cost_function_3(z):
    return (1 + 5 * np.sum(z))**2


# Factory functions
def get_objective_function(name):
    functions = {
        'objective_function_1': objective_function_1,
        'objective_function_2': objective_function_2,
    }
    return functions.get(name)

def get_bounds(name):
    bounds = {
        'objective_function_1': np.array([[0, 1], [0, 1]]), # nx_dim = 2
        'objective_function_2': np.array([[-5, 10], [0, 15]]), # nx_dim = 2
    }
    return bounds.get(name)

# Function to calculate the global minima of the objective function
def calculate_global_minima(obj_func, bounds, noise=0):
    """
    Calculate the global minimum of the objective function.
    
    Parameters:
    - obj_func: The objective function to minimize.
    - bounds: The bounds within which to search for the minimum.
    - noise: Noise level, if applicable.
    
    Returns:
    - global_min: The calculated global minimum value.
    """
    def objective_to_minimize(x):
        return obj_func(x, noise)  # Use highest fidelity and specified noise

    # Use scipy's minimize function with a chosen method (e.g., 'L-BFGS-B')
    result = minimize(objective_to_minimize, x0=bounds[:, 0], bounds=bounds, method='L-BFGS-B')
    
    global_min = result.fun
    return global_min

# Experiment manager function
def run_experiments(config):
    results = {}
    for obj_func_name in config['objective_functions']:
        obj_func = get_objective_function(obj_func_name)
        bounds = get_bounds(obj_func_name)
            
        # Run the optimization:
        x_best, y_best, Xtest_l, ymean_l, ystd_l, Xtrain, ytrain, ndata, i_opt, EI_bool_out = bo.BO_np_scipy(obj_func, bounds, config['budget'], 
                                                                                config['multi_opt'], config['EI_bool'], config['store_data'], config['noise'], config['seed'])
        # Create a dictionary with simulation metadata
        result = {
            'num_iterations run': i_opt+1,
            'n_rs': ndata,
            'EI_bool': EI_bool_out,
            'Xtrain dataset': Xtrain,
            'ytrain dataset': ytrain,
            'x_best': x_best,
            'y_best': y_best
            
            }
        # Store results for each combination
        results[(obj_func_name)] = result
    
    return results

def save_results(results, data_dir):
    # Save the results dictionary as a JSON file
    results_json_path = os.path.join(data_dir, "results.json")
    with open(results_json_path, 'w') as json_file:
        # Convert NumPy arrays to lists for JSON serialization
        results_serializable = {str(k): {key: val.tolist() if isinstance(val, np.ndarray) else val 
                                         for key, val in v.items()} 
                                for k, v in results.items()}
        json.dump(results_serializable, json_file, indent=4)
    
    # Save the results dictionary as a pickle file
    results_pickle_path = os.path.join(data_dir, "results.pkl")
    with open(results_pickle_path, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
    
    # Save the results in CSV format
    # Save the results in CSV format
    for key, result in results.items():
        obj_func_name, cost_func_name = key
        csv_filename = f"{obj_func_name}_{cost_func_name}_results.csv"
        csv_filepath = os.path.join(data_dir, csv_filename)
        
        # Prepare data for DataFrame
        result_dict = {}
        for k, v in result.items():
            # Convert to NumPy array if it's a list
            if isinstance(v, list):
                v = np.array(v)
            
            # Flatten multi-dimensional arrays or handle them appropriately
            if isinstance(v, np.ndarray):
                if v.ndim > 1:
                    v = v.flatten()  # Flatten to 1D if multi-dimensional
                else:
                    v = v  # Keep as is if 1D
            result_dict[k] = v
        
        # Create DataFrame with 1-dimensional data only
        result_df = pd.DataFrame(result_dict)
        
        # Save DataFrame to CSV
        result_df.to_csv(csv_filepath, index=False)

    print(f"Results saved in JSON, pickle, and CSV formats in: {data_dir}")

def plot_results(results):
    # Prompt for a description of the simulation
    simulation_description = input("Please enter a short description of this simulation: ")

    # Create the main directory 'MF_Simulations'
    main_dir = "MF_Simulations"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    
    # Create a subdirectory with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = os.path.join(main_dir, timestamp)
    os.makedirs(sub_dir)
    
    # Create separate directories for plots and data
    plots_dir = os.path.join(sub_dir, "plots")
    data_dir = os.path.join(sub_dir, "data")
    os.makedirs(plots_dir)
    os.makedirs(data_dir)

    # Save the description in a metadata file
    metadata_path = os.path.join(sub_dir, "metadata.txt")
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Simulation Description: {simulation_description}\n")
        metadata_file.write(f"Timestamp: {timestamp}\n")

    # Save the results dictionary
    save_results(results, data_dir)


###########################
##### --- CONTROL --- #####
###########################

if __name__ == "__main__":
    config = {
        'objective_functions': ['objective_function_1', 'objective_function_2'],
        'budget': 2,  # 40, 13, 2
        'multi_opt': 10,
        'EI_bool': False, #Upper Confidence Bound
        'store_data': False, #No need to store_data
        'noise': None,  # either give a value (0.05) or state None
        'seed': 200,  # seed of random number generator
    }

    results = run_experiments(config)
    plot_results(results)
