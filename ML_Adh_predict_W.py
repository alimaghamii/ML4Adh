import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from cpsme.export_figure import export_figure
import joblib
import os
# Here, we import details from utilities
from utilities import load_pb_data, make_predictions
# from utilities import file_path_RPML_5inputs_W
from utilities import file_pathPB
from utilities import save_dir_RPML_5inputs_W
from utilities import process_columns

# Set global styles for font size and axis line width
# Set default font to Times New Roman globally, including math font
font_size =20
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Set default font to Times New Roman
    'mathtext.fontset': 'custom',      # Use custom settings for math text
    'mathtext.rm': 'Times New Roman',  # Roman (non-italic) font
    'mathtext.it': 'Times New Roman:italic',  # Italic font
    'mathtext.bf': 'Times New Roman:bold',    # Bold font
    'axes.labelsize': font_size,              # Font size for labels
    'xtick.labelsize': font_size,             # Font size for x-ticks
    'ytick.labelsize': font_size,             # Font size for y-ticks
    'axes.linewidth': 1,               # Width of the axis lines
    'figure.figsize': (7, 4)          # Default figure size
})

save_dir= save_dir_RPML_5inputs_W
# Ensure the directory exists (optional, if you want to create it if it doesn't exist)
os.makedirs(save_dir, exist_ok=True)

# Create a dictionary to hold the loaded models
loaded_models = {}

# Load each trained model
for model_name in os.listdir(save_dir):
    if model_name.endswith('_model.pkl'):
        model_key = model_name.replace('_model.pkl', '').replace('_', ' ')  # Convert back to original name
        model_path = os.path.join(save_dir, model_name)

        # Load the model and store it in the dictionary
        loaded_models[model_key] = joblib.load(model_path)
        print(f"{model_key} loaded from {model_path}")
    else:
        print(f"Skipped {model_name}, not a model file.")

# Check how many models were loaded
print(f"Total models loaded: {len(loaded_models)}")

# Set fixed values for mu and k
mu_fixed = 3.24
delta_load= 73
delta_l= np.log10(delta_load)
alphabet='a'

k_fixed = - 1  # remember that we put log10 of k in the model
alphaload_fixed = delta_l # * mu_fixed * (np.pi**(2/3))
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 40)  # 100 points between -1 and 6
# n_values = [0.2, 0.4, 0.6, 0.8, 1.6]  # Given n values
n_values = [0.4, 0.6, 1.6]  # Given n values

# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options

# trained_model = models['XGBoost']  # Retrieve trained XGBoost model
trained_model = loaded_models.get('XGBoost')
# trained_model = loaded_models.get('Neural Network')

dataPB = load_pb_data(file_pathPB)

# Plotting the results
fig2 = plt.figure()

predictions = make_predictions(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, n in enumerate(n_values):
    # Use plt.plot() to create a dashed line instead of plt.scatter()
    # plt.scatter(runload_values, predictions[n], label=f'n = {n}', marker='*', s=100, color=colors[i])
    plt.scatter(10** runload_values, 10** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.4,
             lw=2, s= 90)
    plt.plot(10** runload_values, 10** predictions[n], linestyle='-', color=colors[i], label=f'n = {n}', lw=6, alpha= 0.1)


# Add a text arrow
plt.annotate('$n= [0.4, 0.6, 1.6]$', xytext=(10 ** 0, 10 ** 2.25), xy=(10 ** 2, 26),
              arrowprops=dict(facecolor='black', arrowstyle='<-'),
              fontsize=font_size)
plt.text(10**4, 100, f'$\\mu={mu_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(10**4, 65, label, fontsize=font_size)
plt.text(10**5.5, 16, f'({alphabet})', fontsize=font_size)

####################
x_range = np.logspace(2, 6, 100)  # 100 points between 10^4 and 10^6
w_po = 1.1818 * 3.2 * np.pi**(2/3) * (10)**(2/3)
plt.plot(x_range, [w_po] * len(x_range), lw=3,alpha= 0.4, linestyle='--', color='black')
####################

# Adjust x and y axis limits
plt.xlim(1, 10**6)  # Set x-axis limits from 0 to 5
plt.ylim(12, 155)  # Set x-axis limits from 0 to 5

plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{w}_{po}$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
filename = f"RPML_W_vs_r_mu_{mu_fixed}_d_{delta_load}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()
