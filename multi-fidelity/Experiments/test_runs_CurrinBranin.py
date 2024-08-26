import MF_BO_finalgreedy as bo
import dictionaries as dict
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio #pip install numpy matplotlib imageio
import os
import random
from matplotlib import cm
import tkinter as tk
from tkinter import simpledialog
import dictionaries as dict
import json
import os
import pickle
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# Define objective functions

def objective_function_1(x, z, noise=None):
    # Currin exponential (has global maxima, thus -res)
    fracOne = 1 - 0.1*(1-z)*np.exp(-1/(2*x[1] + 1e-8))
    fracTwo = 2300*x[0]**3 + 1900*x[0]**2 + 2092*x[0] + 60
    fracThree = 100*x[0]**3 + 500*x[0]**2 + 4*x[0] + 20
    res = fracOne*fracTwo/fracThree
    if noise is not None:
        mean = 0
        return -res + np.random.normal(mean, np.sqrt(noise))
    else:
        return -res

def objective_function_2(x, z, noise=None):
    # Branin (has global minima, thus res)
    a = 1
    b = 5.1/(4*np.pi**2) - 0.01*(1-z[0])
    c = 5/np.pi - 0.1 * (1-z[1])
    r = 6
    s = 10
    t = 1/(8*np.pi) + 0.05 * (1-z[2])
    res = a*(x[1] - b*x[0] + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s
    if noise is not None:
        mean = 0
        return res + np.random.normal(mean, np.sqrt(noise))
    else:
        return res
    
def objective_function_3(x, z, noise=None):
    #Hartmann 3D

    #nx_dim = 3
    #nz_dim = 4
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


def objective_function_4(x, z, noise=None):
    #Hartmann 6D

    #nx_dim = 6
    #nz_dim = 2
    # Hartmann 6-dimensional parameters
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

#Run code

#x_best, y_best, z_best, c_best, Xtrain, ytrain, ztrain, ctrain, ndata


# Factory functions
def get_objective_function(name):
    functions = {
        'objective_function_1': objective_function_1,
        'objective_function_2': objective_function_2,
        'objective_function_3': objective_function_3,
        'objective_function_4': objective_function_4,
    }
    return functions.get(name)

def get_bounds(name):
    bounds = {
        'objective_function_1': np.array([[0, 1], [0, 1]]), # nx_dim = 2
        'objective_function_2': np.array([[-5, 10], [0, 15]]), # nx_dim = 2
        'objective_function_3': np.array([[0, 1], [0, 1], [0, 1]]), # nx_dim = 3
        'objective_function_4': np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), # nx_dim = 6
    }
    return bounds.get(name)

def get_bounds_fidelity(name):
    bounds = {
        'objective_function_1': np.array([[0.1, 1]]), # nz_dim = 1
        'objective_function_2': np.array([[0, 1], [0, 1], [0, 1]]), # nz_dim = 3
        'objective_function_3': np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), # nx_dim = 4
        'objective_function_4': np.array([[0, 1], [0, 1]]), # nx_dim = 2
    }
    return bounds.get(name)

def get_cost_function(name):
    functions = {
        'cost_function_1': cost_function_1,
        'cost_function_2': cost_function_2,
        'cost_function_3': cost_function_3
    }
    return functions.get(name)

# Experiment manager function
def run_experiments(config):
    results = {}
    for obj_func_name in config['objective_functions']:
        for cost_func_name in config['cost_functions']:
            obj_func = get_objective_function(obj_func_name)
            cost_fun = get_cost_function(cost_func_name)
            bounds = get_bounds(obj_func_name)
            bounds_fid = get_bounds_fidelity(obj_func_name)
            
            # Run the optimization
            for run in range(3):
                result = bo.MFBO_np_scipy(obj_func, cost_fun, bounds, bounds_fid, config['budget'], 
                                      config['multi_opt'], config['noise'], config['seed'], config['kernel_f'])
            
                # Store results for each combination
                results[(obj_func_name, cost_func_name, f'run_{run+1}')] = result
            
                # Optionally save each result to a file
                #save_results_to_file(result, obj_func_name, cost_func_name, run+1)
    
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
    # for key, result in results.items():
    #     obj_func_name, cost_func_name, run = key
    #     csv_filename = f"{obj_func_name}_{cost_func_name}_{run}_results.csv"
    #     csv_filepath = os.path.join(data_dir, csv_filename)
        
    #     # Assuming result is a dictionary with arrays
    #     result_df = pd.DataFrame({k: np.array(v) if isinstance(v, list) else v for k, v in result.items()})
    #     result_df.to_csv(csv_filepath, index=False)

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

    # for (obj_func_name, cost_func_name, run), result in results.items():
    #     # Create a new plot for each objective-cost function combination
    #     plt.figure(figsize=(10, 6))
        
    #     # Set up a colormap for different tones of the same color
    #     colormap = cm.get_cmap('viridis', 3)  # You can choose any colormap and specify the number of runs
        
    #     # Loop through all runs for the current objective-cost combination
    #     for run_num in range(1, 4):  # Assuming you have 3 runs
    #         result_key = (obj_func_name, cost_func_name, f'run_{run_num}')
    #         if result_key in results:
    #             result = results[result_key]
    #             ndata = result['ndata']
    #             y_actual = result['ytrain'][ndata:]
    #             i_opt = np.arange(1, y_actual.shape[0] + 1)  # Iteration numbers
    #             y_max_z = np.zeros_like(y_actual)
    #             Xtrain = result['Xtrain'][ndata:]
    #             bounds_fid = get_bounds_fidelity(obj_func_name)

    #             for i in range(len(i_opt)):
    #                 y_max_z[i] = obj_func_name(Xtrain[i, :], bounds_fid[:, 1], config['noise'])

    #             # Get color from the colormap
    #             color = colormap(run_num - 1)

    #             # Plot y_actual series with a specific marker and varying color tones
    #             plt.plot(i_opt, y_actual, label=f'Actual y value (Run {run_num})', marker='o', color=color)

    #             # Plot y_max_z series with the same color and marker
    #             plt.plot(i_opt, y_max_z, label=f'y value at max z (Run {run_num})', linestyle='--', color=color)
                
    #             # Plot Tom's point
    #             plt.plot(i_opt[-1], result['ynew_Tom'], label=f"Tom's point (Run {run_num})", marker='x', color=color)
        
    #     # Adding titles and labels
    #     plt.title(f'Optimization Progress for {obj_func_name} - {cost_func_name}')
    #     plt.xlabel('Iteration (i_opt)')
    #     plt.ylabel('Objective Function Value (y)')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # Save the plot in the plots subdirectory
    #     plot_filename = f"{obj_func_name}_{cost_func_name}_plot.png"
    #     plot_filepath = os.path.join(plots_dir, plot_filename)
    #     plt.savefig(plot_filepath)
        
    #     # Show the plot
    #     plt.show()


    print(f"Plots and data saved in: {sub_dir}")

###########################
##### --- CONTROL --- #####
###########################

if __name__ == "__main__":
    config = {
        'objective_functions': ['objective_function_1', 'objective_function_2'],
        'cost_functions': ['cost_function_1', 'cost_function_2', 'cost_function_3'],
        'budget': 80, #dim(3 and 4)*10
        'multi_opt': 10,
        'noise': None, # either give a value (0.05) or state None
        'seed': 200, # seed of random number generator
        'kernel_f': True, # Choose from True for "RBF" or False for "Matern5/2"
    }

    results = run_experiments(config)
    plot_results(results)