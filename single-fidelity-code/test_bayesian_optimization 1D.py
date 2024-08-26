import bayesian_optimization as bo
import dictionaries as dict
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio #pip install numpy matplotlib imageio
import os
import datetime
import tkinter as tk
from tkinter import simpledialog

##########################
# --- Test functions --- #
##########################

# x is a row vector
def Rastrigin_f(x, noise=None):
    x = np.array(x)
    x = x - 2  #shift it left cuz 0 sometimes gives issues
    z = 10. + x**2 - 10.*np.cos(2.*np.pi*x)
    if noise is not None:
        mean = 0
        return z + np.random.normal(mean, np.sqrt(noise))
    else:
        return z
# Recommended bounds = np.array([[-3, 3]])

def Ackley_f(x, noise=None):
    x = np.array(x)
    x = x - 2  #shift it left cuz 0 sometimes gives issues
    a = 20.
    b = 0.2
    c = 2.*np.pi

    z = (-a * np.exp(-b*np.sqrt(x**2)) - 
            np.exp(np.cos(c*x)) + a + np.exp(1.))
    if noise is not None:
        mean = 0
        return z + np.random.normal(mean, np.sqrt(noise))
    else:
        return z
# Recommended bounds = np.array([[-32.768 - 2, 32.768 + 2]])

############################
##### --- CONTROLS --- #####
############################

bounds = np.array([[-3, 3]]) #here you determine n_x (x dim.)
iter_tot = 20
multi_opt = 25
EI_bool = False # True = EI, False = UCB
store_data = True # True = gif, False = Plot last iteration
noise = None   # either give a value (0.05) or state None
seed = 200 # seed of random number generator
SaveData = True # save in dictionary or not
function = "Rastrigin_f" # "Rastrigin_f" or "Ackley_f"

if function == "Rastrigin_f":
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

if store_data == True:
    # Create a directory to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')

    filenames = []

    iter = ymean_l.shape[0] #i_opt where i cut, it's from 0 to i_opt - 1

    for i in range(iter):
        plt.figure(figsize=(20, 12))
        # plot observed points
        row = ndata + i
        plt.plot(Xtrain[:row], ytrain[:row], 'kx', mew=2)
        # plot the samples of posteriors
        plt.plot(Xtest_l, fun(Xtest_l, noise), 'black', linewidth=1)
        # plot GP mean
        plt.plot(Xtest_l, ymean_l[i,:], 'C0', lw=2)
        # plot GP confidence intervals
        plt.fill_between(Xtest_l.flatten(), 
                     ymean_l[i] - 2 * ystd_l[i], 
                     ymean_l[i] + 2 * ystd_l[i],
                     color='C0', alpha=0.2)
        # plot the new point being added
        if i > 0:
            plt.axvline(x=Xtrain[row-1], color='red', linestyle='--')
        plt.title(f'Iteration {i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(('training', 'true function', 'GP mean', 'GP conf interval'),
           loc='lower right')
        filename = f'plots/plot_{i}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)

    # Total duration for the GIF
    total_duration = 15  # seconds
    # Calculate the number of frames required for 10 seconds at 10 FPS
    fps = 10
    total_frames = total_duration * fps

    # Calculate the number of times each frame should be repeated
    frame_repeats = total_frames // (iter_tot+1)

    gif_name = f"BO_{function}.gif"
    # Create the GIF
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            for _ in range(frame_repeats):
                writer.append_data(image)

    # Cleanup: Remove the plot images after creating the GIF
    for filename in filenames:
        os.remove(filename)

    print("GIF created successfully.")
    
    if SaveData ==True:
        # Prompt the user for a text message describing the run
        #text_message = input("Please enter a description of this simulation run: ")
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
            folder_name = os.path.join(simulation_runs_directory, f"{function}1D_{timestamp}")
    
            # Save all the data
            dict.save_simulation_data_gif(simulation_data, gif_name, description, folder_name)
    
            # Close the plots to free up memory
            plt.close('all')
        else:
            print("No description entered. Simulation aborted.")
else:
    # --- Plotting last iteration --- #
    fig, ax = plt.subplots()

    # Plot observed points
    ax.plot(Xtrain, ytrain, 'kx', mew=2)

    # Plot the true function
    ax.plot(Xtest_l, fun(Xtest_l, noise), 'black', linewidth=1)

    # Plot GP confidence intervals
    ax.fill_between(Xtest_l.flatten(), 
                ymean_l - 2 * ystd_l, 
                ymean_l + 2 * ystd_l, 
                color='C0', alpha=0.2)

    # Plot GP mean
    ax.plot(Xtest_l, ymean_l, 'C0', lw=2)

    # Set axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-15, 45)

    # Set plot title
    ax.set_title('Gaussian Process Regression')

    # Add legend
    ax.legend(['training', 'true function', 'GP conf interval', 'GP mean'], loc='lower right')

    # Display the plot
    plt.show()

    
    if SaveData ==True:
        # Save plots
        plots = []
        plots.append(fig)

        # Prompt the user for a text message describing the run
        #text_message = input("Please enter a description of this simulation run: ")
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
            folder_name = os.path.join(simulation_runs_directory, f"{function}1D_{timestamp}")
    
            # Save all the data
            dict.save_simulation_data(simulation_data, plots, description, folder_name)
    
            # Close the plots to free up memory
            plt.close('all')
        else:
            print("No description entered. Simulation aborted.")
