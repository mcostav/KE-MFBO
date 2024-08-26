import bayesian_optimization as bo
import dictionaries as dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import json
import os
import datetime
import tkinter as tk
from tkinter import simpledialog

##########################
# --- Test functions --- #
##########################

# x is a row vector
def Rosenbrock_f(x, noise=None):
    x = np.array(x)
    n = len(x)
    z = 0
    for i in range(n-1):
        z += (x[i] - 1)**2 + 100 * (x[i+1] - x[i]**2)**2
    if noise is not None:
        mean = 0
        return z + np.random.normal(mean, np.sqrt(noise))
    else:
        return z
# Recommended bounds = np.array([[-3.0, 4.0], [-3.0, 4.0]])

def Levy_f(x):
    n_x = x.shape[1]
    x = x.reshape((n_x, 1))
    w = 1. + (x-1.)/4.
    z = np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1.)**2*(1. + 10.*np.sin(np.pi*w[:-1]+1.)**2)) \
        + (w[-1]-1.)**2*(1.+np.sin(2.*np.pi*w[-1]))
    return z
# Recommended bounds = np.array([[-10, 10], [-10, 10]])

def Rastrigin_f(x):
    n_x = x.shape[1]
    x = x.reshape((n_x, 1))
    z = 10.*n_x + np.sum(x**2 - 10.*np.cos(2.*np.pi*x))
    return z    #check but this should work well, I revised it to make sure dimensions make sense!
# Recommended bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])

def Ackley_f(x):
    n_x = x.shape[1]
    a = 20.
    b = 0.2
    c = 2.*np.pi

    x = x.reshape((n_x, 1))
    z = (-a * np.exp(-b*np.sqrt(1./n_x*np.sum(x**2))) - 
            np.exp(1./n_x*np.sum(np.cos(c*x))) + a + np.exp(1.))
    return z
# Recommended bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]])

############################
##### --- CONTROLS --- #####
############################

bounds = np.array([[-6.0, 6.0], [-6.0, 6.0]]) #here you determine n_x (x dim.)
iter_tot = 20
multi_opt = 25
EI_bool = False # True = EI, False = UCB
store_data = True # True = Store all iterations, False = Plot last iteration
noise = None # either give a value (0.05) or state None
seed = 42 # seed of random number generator
SaveData = True # save in dictionary or not
function = "Rosenbrock_f" # "Rosenbrock_f" or "Levy_f" or "Rastrigin_f" or "Ackley_f"

if function == "Rosenbrock_f":
    fun = Rosenbrock_f
elif function == "Levy_f":
    fun = Levy_f
elif function == "Rastrigin_f":
    fun = Rastrigin_f
elif function == "Ackley_f":
        fun = Ackley_f
else:
    print(f'Introduce a valid coded function')

#Run code

x_best, y_best, Xtest_l, ymean_l, ystd_l, Xtrain, ytrain, ndata, i_opt = bo.BO_np_scipy(fun, bounds, 
                                                                          iter_tot, multi_opt, EI_bool, store_data, noise, seed)
#Print some results
print("BO x best value: ", x_best)
print("BO y best value: ", y_best)
print(f'Xtrain:', Xtrain)
print(f'ytrain:', ytrain)

##############################
### --- Data treatment --- ###
##############################

if SaveData == True:
        # Create a dictionary with simulation metadata
        simulation_data = {
            "num_iterations run": i_opt+1,
            "objective_function": function,
            "acquisition function": EI_bool,
            "bounds": bounds.tolist(),
            "seed": seed,
            "Xtrain dataset": Xtrain.tolist(),
            "ytrain dataset": ytrain.tolist(),
            "x_best": x_best.tolist(),
            "y_best": y_best.tolist()
            
        }

#x = np.linspace(-2.0, 2.0, 500)
#y = np.linspace(-2.0, 2.0, 500)
#X, Y = np.meshgrid(x,y)
X, Y = np.meshgrid(Xtest_l[:,0], Xtest_l[:,1])
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = fun(np.hstack([X[i, j], Y[i, j]]))
    
# Plot with color-filled contours
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the function contour with filled colors
contour_filled = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis)

# Add contour lines on top of the filled contours
contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5)

# Add labels to contour lines
ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

# Add a few contours around zero
contour_levels = [-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5]
contour_lines = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.5)

# Label the contour lines
ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')

# Plot the selected points
ax.scatter(Xtrain[:, 0], Xtrain[:, 1], color='red', s=50, label='Selected Points', marker='x')

# Adding a color bar for the contour
cbar = plt.colorbar(contour_filled)
cbar.set_label('Objective Function Value')

ax.set_title("Objective Function with Filled Contour and Selected Points")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

plt.show()

if SaveData == True:
    # Save plots
    plots = []
    plots.append(fig)

    # Create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt for description
    description = simpledialog.askstring("Input", "Please enter a description of this simulation run:")

    # Run the simulation with the provided description
    if description:
    
        # Get the directory where the current script is located
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the Simulation_Runs directory
        simulation_runs_directory = os.path.join(script_directory, "Simulation_Runs")
    
        # Create the Simulation_Runs directory if it doesn't exist
        os.makedirs(simulation_runs_directory, exist_ok=True)
    
        # Create a folder name based on the function and timestamp within Simulation_Runs
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(simulation_runs_directory, f"{function}2D_{timestamp}")
    
        # Save all the data
        dict.save_simulation_data(simulation_data, plots, description, folder_name)
    
        # Close the plots to free up memory
        plt.close('all')
    else:
        print("No description entered. Simulation aborted.")