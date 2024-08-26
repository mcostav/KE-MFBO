import os
import json

###################################################
## --- Functions to save files in dictionary --- ##
###################################################

# Function to save simulation data and files
def save_simulation_data_gif(simulation_data, gif_name, text_message, folder_name):
    # Create a directory for the simulation
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the dictionary as a JSON file
    with open(os.path.join(folder_name, 'simulation_data.json'), 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)
        
    # Move the GIF to the folder
    gif_dest_path = os.path.join(folder_name, gif_name)
    os.rename(gif_name, gif_dest_path)
    
    # Save the text message
    with open(os.path.join(folder_name, 'description.txt'), 'w') as json_file:
        json_file.write(text_message)
    
    print(f"Simulation data saved in {folder_name}")

def save_simulation_data(simulation_data, plots, text_message, folder_name):
    # Create a directory for the simulation
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the dictionary as a JSON file
    with open(os.path.join(folder_name, 'simulation_data.json'), 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)
    
    # Save each plot
    for i, plot in enumerate(plots):
        plot_path = os.path.join(folder_name, f'plot_{i+1}.png')
        plot.savefig(plot_path)
    
    # Save the text message
    with open(os.path.join(folder_name, 'description.txt'), 'w') as json_file:
        json_file.write(text_message)
    
    print(f"Simulation data saved in {folder_name}")