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
    result = minimize(objective_to_minimize, x0=bounds[:, 1], bounds=bounds, method='L-BFGS-B')
    
    global_min = result.fun
    return global_min

def load_results_from_json(data_dir):
    """
    Load the results from a JSON file.
    
    Parameters:
    - data_dir: The directory where the results files are stored.
    
    Returns:
    - results: The loaded results dictionary.
    """
    results_json_path = os.path.join(data_dir, "results_Hartmann_noiseless.json")
    with open(results_json_path, 'r') as json_file:
        results_json = json.load(json_file)
        # Convert lists back to numpy arrays where necessary
        results = {tuple(eval(k)): {key: np.array(val) if isinstance(val, list) else val 
                                    for key, val in v.items()}
                   for k, v in results_json.items()}
    
    return results

def load_results_from_pkl(data_dir):
    """
    Load the results from a PKL file.
    
    Parameters:
    - data_dir: The directory where the results files are stored.
    
    Returns:
    - results: The loaded results dictionary.
    """
    results_pickle_path = os.path.join(data_dir, "results.pkl")
    with open(results_pickle_path, 'rb') as pickle_file:
        results = pickle.load(pickle_file)
    
    return results

def plot_results_from_files(data_dir, config, use_json=True):
    # Load the results from either JSON or PKL file
    if use_json:
        results = load_results_from_json(data_dir)
    else:
        results = load_results_from_pkl(data_dir)
    
    # Proceed with the existing plotting mechanism
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
    os.makedirs(plots_dir)

    # Save the description in a metadata file
    metadata_path = os.path.join(sub_dir, "metadata.txt")
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Simulation Description: {simulation_description}\n")
        metadata_file.write(f"Timestamp: {timestamp}\n")

    # Calculate the global minima for each objective function used in the results
    global_minima = {}
    for obj_func_name, _ in set((key[0], key[1]) for key in results.keys()):
        obj_func = get_objective_function(obj_func_name)
        bounds = get_bounds(obj_func_name)
        bounds_fid = get_bounds_fidelity(obj_func_name)
        global_minima[obj_func_name] = calculate_global_minima(obj_func, bounds, bounds_fid, config['noise'])

    for (obj_func_name, cost_func_name), result in results.items():
        # Retrieve the global minimum for the current objective function
        global_min = global_minima[obj_func_name]

        # Create a new plot for each objective-cost function combination
        plt.figure(figsize=(10, 6))
        
        result_key = (obj_func_name, cost_func_name)
        if result_key in results:
            result = results[result_key]
            ndata = result['ndata']
            y_actual = result['ytrain'][ndata:]
            i_opt = np.arange(1, y_actual.shape[0] + 1)  # Iteration numbers
            y_max_z = np.zeros_like(y_actual)
            Xtrain = result['Xtrain'][ndata:]
            ztrain = result['ztrain'][ndata:]  # Extract ztrain
            bounds_fid = get_bounds_fidelity(obj_func_name)

            for i in range(len(i_opt)):
                y_max_z[i] = get_objective_function(obj_func_name)(Xtrain[i, :], bounds_fid[:, 1], config['noise'])

            # Plot y_actual and y_max_z with markers
            plt.plot(i_opt, y_actual, label=f'Actual y value', marker='x', linestyle='-', color='C0')
            plt.plot(i_opt, y_max_z, label=f'y value at max z', linestyle='--', marker='x', color='C1')
                
            # Plot Savage et. al. x_greedy point
            plt.plot(i_opt[-1], result['ynew_Tom'], label="Savage et. al. x_greedy", marker='x', color='C2')

            # Highlight points at highest fidelity (z = 1) with red circles
            high_fidelity_indices = np.all(ztrain == 1, axis=1)  # Check if all fidelity dimensions are 1
            plt.scatter(i_opt[high_fidelity_indices], y_actual[high_fidelity_indices], facecolors='none', edgecolors='red', s=100)
            plt.scatter(i_opt[high_fidelity_indices], y_max_z[high_fidelity_indices], facecolors='none', edgecolors='red', s=100, label='Highest Fidelity')

        # Plot the global minimum as a horizontal green dotted line
        plt.axhline(y=global_min, color='green', linestyle=':', label='Global Minima')

        # Adding titles and labels
        plt.title(f'Optimization Progress for {obj_func_name} - {cost_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Objective Function Value (y)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_{cost_func_name}_plot.png"))
        plt.show()

        # --- Simple Regret Plot Section ---
        # Calculate simple regret for each run
        simple_regret = abs(global_min - np.minimum.accumulate(y_actual))
        
        # Plotting Simple Regret with green color
        plt.figure(figsize=(10, 6))
        plt.plot(i_opt, simple_regret, label=f'Simple Regret', linestyle='-', color='green')
        
        # Adding titles and labels for the simple regret plot
        plt.title(f'Simple Regret for {obj_func_name} - {cost_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Simple Regret')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_{cost_func_name}_simple_regret_plot.png"))
        plt.show()

        # --- Updated Plot: Plotting ztrain (Fidelities) with horizontal and vertical lines ---
        plt.figure(figsize=(10, 6))
        for i in range(ztrain.shape[1]):
            # Create horizontal lines for each iteration
            for j in range(len(i_opt) - 1):
                plt.plot([i_opt[j], i_opt[j+1]], [ztrain[j, i], ztrain[j, i]], '-', color=f'C{i}', linewidth=2)
                plt.plot([i_opt[j+1], i_opt[j+1]], [ztrain[j, i], ztrain[j+1, i]], '-', color=f'C{i}', linewidth=2)
            # Add the last point
            plt.plot([i_opt[-1], i_opt[-1]], [ztrain[-1, i], ztrain[-1, i]], '-', color=f'C{i}', linewidth=2, label=f'Fidelity z[{i+1}]')

        # Restrict the y-axis to [0, 1] and adjust to ensure visibility of z = 1 line
        plt.ylim(-0.05, 1.05)

        # Adding titles and labels for the fidelities plot
        plt.title(f'Selected Fidelities for {obj_func_name} - {cost_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Fidelity Value')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_{cost_func_name}_fidelities_plot.png"))
        plt.show()

        # --- New Plot: Minimum Euclidean Distance with green color ---
        min_distances = []
        for i in range(1, len(y_actual)):
            current_point = y_actual[i]
            previous_points = y_actual[:i]
            min_distance = np.min([np.linalg.norm(current_point - prev_point) for prev_point in previous_points])
            min_distances.append(min_distance)

        plt.figure(figsize=(10, 6))
        plt.plot(i_opt[1:], min_distances, label=f'Min Euclidean Distance', linestyle='-', color='green')
        
        # Adding titles and labels for the min distance plot
        plt.title(f'Minimum Euclidean Distance for {obj_func_name} - {cost_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Min Euclidean Distance')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_{cost_func_name}_min_distance_plot.png"))
        plt.show()

    print(f"Plots and data saved in: {sub_dir}")


    
    

if __name__ == "__main__":
    config = {
        'objective_functions': ['objective_function_1', 'objective_function_2'],
        'cost_functions': ['cost_function_1', 'cost_function_2', 'cost_function_3'],
        'budget': 30,
        'multi_opt': 10,
        'noise': None,
        'seed': 200,
        'kernel_f': True,
    }
    
    data_directory = "."  # Set to the current directory
    use_json = True  # Set to False if you prefer to load from PKL instead
    
    plot_results_from_files(data_directory, config, use_json)