# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import json
# from datetime import datetime
# from scipy.optimize import minimize

# # Define objective functions
# def objective_function_1(x, noise=None):
#     fracOne = 1
#     fracTwo = 2300 * x[0]**3 + 1900 * x[0]**2 + 2092 * x[0] + 60
#     fracThree = 100 * x[0]**3 + 500 * x[0]**2 + 4 * x[0] + 20
#     res = fracOne * fracTwo / fracThree
#     if noise is not None:
#         mean = 0
#         return -res + np.random.normal(mean, np.sqrt(noise))
#     else:
#         return -res

# def objective_function_2(x, noise=None):
#     a = 1
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6
#     s = 10
#     t = 1 / (8 * np.pi)
#     res = a * (x[1] - b * x[0] + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s
#     if noise is not None:
#         mean = 0
#         return res + np.random.normal(mean, np.sqrt(noise))
#     else:
#         return res

# # Factory functions
# def get_objective_function(name):
#     functions = {
#         'objective_function_1': objective_function_1,
#         'objective_function_2': objective_function_2,
#     }
#     return functions.get(name)

# def get_bounds(name):
#     bounds = {
#         'objective_function_1': np.array([[0, 1], [0, 1]]),  # nx_dim = 2
#         'objective_function_2': np.array([[-5, 10], [0, 15]]),  # nx_dim = 2
#     }
#     return bounds.get(name)

# # Function to calculate the global minima of the objective function
# def calculate_global_minima(obj_func, bounds, noise=0):
#     def objective_to_minimize(x):
#         return obj_func(x, noise)

#     result = minimize(objective_to_minimize, x0=bounds[:, 0], bounds=bounds, method='L-BFGS-B')
#     global_min = result.fun
#     return global_min

# def load_results_from_json(file_path):
#     with open(file_path, 'r') as json_file:
#         results_json = json.load(json_file)
#         return results_json

# def plot_combined_results_for_objectives(file_paths, config):
#     results_list = [load_results_from_json(fp) for fp in file_paths]

#     simulation_description = input("Please enter a short description of this simulation: ")

#     main_dir = "MF_Simulations"
#     if not os.path.exists(main_dir):
#         os.makedirs(main_dir)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     sub_dir = os.path.join(main_dir, timestamp)
#     os.makedirs(sub_dir)
    
#     plots_dir = os.path.join(sub_dir, "plots")
#     os.makedirs(plots_dir)

#     metadata_path = os.path.join(sub_dir, "metadata.txt")
#     with open(metadata_path, 'w') as metadata_file:
#         metadata_file.write(f"Simulation Description: {simulation_description}\n")
#         metadata_file.write(f"Timestamp: {timestamp}\n")

#     for obj_func_name in config['objective_functions']:
#         obj_func = get_objective_function(obj_func_name)
#         bounds = get_bounds(obj_func_name)
#         global_min = calculate_global_minima(obj_func, bounds, config['noise'])

#         # Determine the maximum number of iterations from the low-cost file
#         max_iterations = len(results_list[0][obj_func_name]['ytrain dataset'])  # Assuming this is the maximum
#         i_opt = np.arange(1, max_iterations + 1)

#         # Plot y_actual for all datasets
#         plt.figure(figsize=(10, 6))
#         for idx, results in enumerate(results_list):
#             if obj_func_name not in results:
#                 continue
            
#             result = results[obj_func_name]
#             ndata = result['n_rs']
#             y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

#             # Adjust lengths by padding with NaNs if necessary
#             if len(y_actual) < max_iterations:
#                 y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

#             plt.plot(i_opt[:len(y_actual)], y_actual, label=f'Actual y value - Dataset {idx + 1}', marker='x', linestyle='-', color=f'C{idx}')

#         plt.axhline(y=global_min, color='green', linestyle=':', label='Global Minima')
#         plt.title(f'Optimization Progress for {obj_func_name}')
#         plt.xlabel('Iteration (i_opt)')
#         plt.ylabel('Objective Function Value (y)')
#         plt.legend()
#         plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_y_actual_plot_combined.png"))
#         plt.show()

#         # Plot Simple Regret for all datasets
#         plt.figure(figsize=(10, 6))
#         for idx, results in enumerate(results_list):
#             if obj_func_name not in results:
#                 continue
            
#             result = results[obj_func_name]
#             y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

#             # Adjust lengths by padding with NaNs if necessary
#             if len(y_actual) < max_iterations:
#                 y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

#             valid_y_actual = y_actual[~np.isnan(y_actual)]
#             simple_regret = np.abs(global_min - np.fmin.accumulate(valid_y_actual))

#             plt.plot(i_opt[:len(simple_regret)], simple_regret, label=f'Simple Regret - Dataset {idx + 1}', linestyle='-', color=f'C{idx}')
        
#         plt.title(f'Simple Regret for {obj_func_name}')
#         plt.xlabel('Iteration (i_opt)')
#         plt.ylabel('Simple Regret')
#         plt.legend()
#         plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_combined_simple_regret_plot.png"))
#         plt.show()

#         # Plot Minimum Euclidean Distance for all datasets
#         plt.figure(figsize=(10, 6))
#         for idx, results in enumerate(results_list):
#             if obj_func_name not in results:
#                 continue
            
#             result = results[obj_func_name]
#             y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

#             # Adjust lengths by padding with NaNs if necessary
#             if len(y_actual) < max_iterations:
#                 y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

#             valid_y_actual = y_actual[~np.isnan(y_actual)]
#             min_distances = []
#             for i in range(1, len(valid_y_actual)):
#                 current_point = valid_y_actual[i]
#                 previous_points = valid_y_actual[:i]
#                 if len(previous_points) > 0:
#                     min_distance = np.min([np.linalg.norm(current_point - prev_point) for prev_point in previous_points])
#                     min_distances.append(min_distance)

#             plt.plot(i_opt[1:len(min_distances) + 1], min_distances, label=f'Min Euclidean Distance - Dataset {idx + 1}', linestyle='-', color=f'C{idx}')
        
#         plt.title(f'Minimum Euclidean Distance for {obj_func_name}')
#         plt.xlabel('Iteration (i_opt)')
#         plt.ylabel('Min Euclidean Distance')
#         plt.legend()
#         plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_combined_min_distance_plot.png"))
#         plt.show()

#         # Plot Fidelity Evolution for all datasets
#         plt.figure(figsize=(10, 6))
#         for idx, results in enumerate(results_list):
#             if obj_func_name not in results:
#                 continue
            
#             result = results[obj_func_name]
#             ztrain = np.array(result['Xtrain dataset'][ndata:])
            
#             # Ensure we have a consistent length for plotting
#             if len(ztrain) < max_iterations:
#                 ztrain = np.pad(ztrain, ((0, max_iterations - len(ztrain)), (0, 0)), constant_values=np.nan)

#             for j in range(ztrain.shape[1]):
#                 plt.plot(i_opt, ztrain[:, j], label=f'Fidelity z[{j+1}] - Dataset {idx + 1}', linestyle='-', marker='x', color=f'C{idx*2+j}')

#         plt.ylim(-0.05, 1.05)
#         plt.title(f'Fidelity Evolution for {obj_func_name}')
#         plt.xlabel('Iteration (i_opt)')
#         plt.ylabel('Fidelity Value')
#         plt.legend()
#         plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_combined_fidelity_plot.png"))
#         plt.show()

#     print(f"Plots and data saved in: {sub_dir}")

# if __name__ == "__main__":
#     config = {
#         'objective_functions': ['objective_function_1', 'objective_function_2'],
#         'budget': 2,  # 40, 13, 2
#         'multi_opt': 10,
#         'EI_bool': False,  # Upper Confidence Bound
#         'store_data': False,  # No need to store_data
#         'noise': None,  # either give a value (0.05) or state None
#         'seed': 200,  # seed of random number generator
#     }

#     # List your JSON file paths here
#     file_paths = [
#         "lowCostf_singlefid.json",  # Replace with your actual file paths
#         "mediumCostf_singlefid.json",
#         "highCostf_singlefid.json"
#     ]

#     plot_combined_results_for_objectives(file_paths, config)

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from scipy.optimize import minimize

# Define objective functions
def objective_function_1(x, noise=None):
    fracOne = 1
    fracTwo = 2300 * x[0]**3 + 1900 * x[0]**2 + 2092 * x[0] + 60
    fracThree = 100 * x[0]**3 + 500 * x[0]**2 + 4 * x[0] + 20
    res = fracOne * fracTwo / fracThree
    if noise is not None:
        mean = 0
        return -res + np.random.normal(mean, np.sqrt(noise))
    else:
        return -res

def objective_function_2(x, noise=None):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    res = a * (x[1] - b * x[0] + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s
    if noise is not None:
        mean = 0
        return res + np.random.normal(mean, np.sqrt(noise))
    else:
        return res

# Factory functions
def get_objective_function(name):
    functions = {
        'objective_function_1': objective_function_1,
        'objective_function_2': objective_function_2,
    }
    return functions.get(name)

def get_bounds(name):
    bounds = {
        'objective_function_1': np.array([[0, 1], [0, 1]]),  # nx_dim = 2
        'objective_function_2': np.array([[-5, 10], [0, 15]]),  # nx_dim = 2
    }
    return bounds.get(name)

# Function to calculate the global minima of the objective function
def calculate_global_minima(obj_func, bounds, noise=0):
    def objective_to_minimize(x):
        return obj_func(x, noise)

    result = minimize(objective_to_minimize, x0=bounds[:, 0], bounds=bounds, method='L-BFGS-B')
    global_min = result.fun
    return global_min

def load_results_from_json(file_path):
    with open(file_path, 'r') as json_file:
        results_json = json.load(json_file)
        return results_json

def plot_combined_results_for_objectives(file_paths, config):
    results_list = [load_results_from_json(fp) for fp in file_paths]

    simulation_description = input("Please enter a short description of this simulation: ")

    main_dir = "MF_Simulations"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = os.path.join(main_dir, timestamp)
    os.makedirs(sub_dir)
    
    plots_dir = os.path.join(sub_dir, "plots")
    os.makedirs(plots_dir)

    metadata_path = os.path.join(sub_dir, "metadata.txt")
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Simulation Description: {simulation_description}\n")
        metadata_file.write(f"Timestamp: {timestamp}\n")

    for obj_func_name in config['objective_functions']:
        obj_func = get_objective_function(obj_func_name)
        bounds = get_bounds(obj_func_name)
        global_min = calculate_global_minima(obj_func, bounds, config['noise'])

        # Determine the maximum number of iterations from the low-cost file
        max_iterations = len(results_list[0][obj_func_name]['ytrain dataset'])  # Assuming this is the maximum
        i_opt = np.arange(1, max_iterations + 1)

        # Plot y_actual for all datasets
        plt.figure(figsize=(10, 6))
        for idx, results in enumerate(results_list):
            if obj_func_name not in results:
                continue
            
            result = results[obj_func_name]
            ndata = result['n_rs']
            y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

            # Adjust lengths by padding with NaNs if necessary
            if len(y_actual) < max_iterations:
                y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

            plt.plot(i_opt[:len(y_actual)], y_actual, label=f'Actual y value - Dataset {idx + 1}', marker='x', linestyle='-', color=f'C{idx}')

        plt.axhline(y=global_min, color='green', linestyle=':', label='Global Minima')
        plt.title(f'Optimization Progress for {obj_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Objective Function Value (y)')
        plt.legend(loc='lower right')  # Place the legend in the lower right corner
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_y_actual_plot_combined.png"))
        plt.show()

        # Plot Simple Regret for all datasets
        plt.figure(figsize=(10, 6))
        for idx, results in enumerate(results_list):
            if obj_func_name not in results:
                continue
            
            result = results[obj_func_name]
            y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

            # Adjust lengths by padding with NaNs if necessary
            if len(y_actual) < max_iterations:
                y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

            valid_y_actual = y_actual[~np.isnan(y_actual)]
            simple_regret = np.abs(global_min - np.fmin.accumulate(valid_y_actual))

            plt.plot(i_opt[:len(simple_regret)], simple_regret, label=f'Simple Regret - Dataset {idx + 1}', linestyle='-', color=f'C{idx}')
        
        plt.title(f'Simple Regret for {obj_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Simple Regret')
        plt.legend(loc='lower right')  # Place the legend in the lower right corner
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_combined_simple_regret_plot.png"))
        plt.show()

        # Plot Minimum Euclidean Distance for all datasets
        plt.figure(figsize=(10, 6))
        for idx, results in enumerate(results_list):
            if obj_func_name not in results:
                continue
            
            result = results[obj_func_name]
            y_actual = np.array(result['ytrain dataset'][ndata:]).flatten()

            # Adjust lengths by padding with NaNs if necessary
            if len(y_actual) < max_iterations:
                y_actual = np.pad(y_actual, (0, max_iterations - len(y_actual)), constant_values=np.nan)

            valid_y_actual = y_actual[~np.isnan(y_actual)]
            min_distances = []
            for i in range(1, len(valid_y_actual)):
                current_point = valid_y_actual[i]
                previous_points = valid_y_actual[:i]
                if len(previous_points) > 0:
                    min_distance = np.min([np.linalg.norm(current_point - prev_point) for prev_point in previous_points])
                    min_distances.append(min_distance)

            plt.plot(i_opt[1:len(min_distances) + 1], min_distances, label=f'Min Euclidean Distance - Dataset {idx + 1}', linestyle='-', color=f'C{idx}')
        
        plt.title(f'Minimum Euclidean Distance for {obj_func_name}')
        plt.xlabel('Iteration (i_opt)')
        plt.ylabel('Min Euclidean Distance')
        plt.legend(loc='lower right')  # Place the legend in the lower right corner
        plt.savefig(os.path.join(plots_dir, f"{obj_func_name}_combined_min_distance_plot.png"))
        plt.show()

    print(f"Plots and data saved in: {sub_dir}")

if __name__ == "__main__":
    config = {
        'objective_functions': ['objective_function_1', 'objective_function_2'],
        'budget': 2,  # 40, 13, 2
        'multi_opt': 10,
        'EI_bool': False,  # Upper Confidence Bound
        'store_data': False,  # No need to store_data
        'noise': None,  # either give a value (0.05) or state None
        'seed': 200,  # seed of random number generator
    }

    # List your JSON file paths here
    file_paths = [
        "lowCostf_singlefid.json",  # Replace with your actual file paths
        "mediumCostf_singlefid.json",
        "highCostf_singlefid.json"
    ]

    plot_combined_results_for_objectives(file_paths, config)


