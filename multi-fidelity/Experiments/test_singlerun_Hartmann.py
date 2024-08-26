import MF_BO_finalgreedy as bo
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
    
def objective_function_1(x, z, noise=None):
    #Hartmann 3D (has global maxima, thus -res)
    #Parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10.0, 30.0], 
                  [0.1, 10.0, 35.0], 
                  [3.0, 10.0, 30.0], 
                  [0.1, 10.0, 35.0]])
    P = 1e-4 * np.array([[3689, 1170, 2673], 
                         [4699, 4387, 7470], 
                         [1091, 8732, 5547], 
                         [381, 5743, 8828]])
    
    # Calculate the modified alpha based on fidelity z
    alpha_prime = np.sum(0.1 * (1 - z))  # Alpha' modification
    alpha_modified = alpha - alpha_prime  # Adjusted alpha for fidelity

    # Calculate the function value
    outer = 0.0
    for i in range(4):
        alpha_modified = 0.1 * (1-z[i])
        inner = 0.0
        for j in range(3):
            inner += A[i, j] * (x[j] - P[i, j]) ** 2
        outer += (alpha[i] - alpha_modified) * np.exp(-inner)

    if noise is not None:
        mean = 0
        return -outer + np.random.normal(mean, np.sqrt(noise))
    else:
        return -outer


def objective_function_2(x, z, noise=None):
    #Hartmann 6D (has global maxima, thus -res)
    #Parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], 
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0], 
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], 
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], 
                         [2329, 4135, 8307, 3736, 1004, 9991], 
                         [2348, 1451, 3522, 2883, 3047, 6650], 
                         [4047, 8828, 8732, 5743, 1091, 381]])
    
    # Calculate the modified alpha based on fidelity z
    alpha_prime = np.sum(0.1 * (1 - z))  # Alpha' modification
    alpha_modified = alpha - alpha_prime  # Adjusted alpha for fidelity

    # Calculate the function value
    outer = 0.0
    for i in range(2):
        alpha_modified = 0.1 * (1-z[i])
        inner = 0.0
        for j in range(6):
            inner += A[i, j] * (x[j] - P[i, j]) ** 2
        outer += (alpha[i] - alpha_modified) * np.exp(-inner)

    if noise is not None:
        mean = 0
        return -outer + np.random.normal(mean, np.sqrt(noise))
    else:
        return -outer

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
        'objective_function_1': np.array([[0, 1], [0, 1], [0, 1]]), # nx_dim = 3
        'objective_function_2': np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), # nx_dim = 6
    }
    return bounds.get(name)

def get_bounds_fidelity(name):
    bounds = {
        'objective_function_1': np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), # nx_dim = 4
        'objective_function_2': np.array([[0, 1], [0, 1]]), # nx_dim = 2
    }
    return bounds.get(name)

def get_cost_function(name):
    functions = {
        'cost_function_1': cost_function_1,
        'cost_function_2': cost_function_2,
        'cost_function_3': cost_function_3
    }
    return functions.get(name)

# Function to calculate the global minima of the objective function
def calculate_global_minima(obj_func, bounds, bounds_fid, noise=0):
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
        return obj_func(x, bounds_fid[:, 1], noise)  # Use highest fidelity and specified noise

    # Use scipy's minimize function with a chosen method (e.g., 'L-BFGS-B')
    result = minimize(objective_to_minimize, x0=bounds[:, 0], bounds=bounds, method='L-BFGS-B')
    
    global_min = result.fun
    return global_min

# Experiment manager function
def run_experiments(config):
    results = {}
    for obj_func_name in config['objective_functions']:
        for cost_func_name in config['cost_functions']:
            obj_func = get_objective_function(obj_func_name)
            cost_fun = get_cost_function(cost_func_name)
            bounds = get_bounds(obj_func_name)
            bounds_fid = get_bounds_fidelity(obj_func_name)
            
            # Run the optimization:
            result = bo.MFBO_np_scipy(obj_func, cost_fun, bounds, bounds_fid, config['budget'], 
                                      config['multi_opt'], config['noise'], config['seed'], config['kernel_f'])
            
            # Store results for each combination
            results[(obj_func_name, cost_func_name)] = result
    
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
        'cost_functions': ['cost_function_1', 'cost_function_2', 'cost_function_3'],
        'budget': 200,  # dim(7-8)*10
        'multi_opt': 10,
        'noise': None,  # either give a value (0.05) or state None
        'seed': 200,  # seed of random number generator
        'kernel_f': True,  # Choose from True for "RBF" or False for "Matern5/2"
    }

    results = run_experiments(config)
    plot_results(results)
