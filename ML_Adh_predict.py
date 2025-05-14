import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from cpsme.export_figure import export_figure
import joblib
import os
# Here, we import details from utilities
from utilities import load_pb_data, make_predictions
from utilities import file_path_RPML_5inputs
from utilities import file_pathPB
from utilities import save_dir_RPML_5inputs
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

save_dir= save_dir_RPML_5inputs
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
figure = 7 # specify the figure that you want to creat
alphabet = 'a'

if figure== 7:
    if alphabet=='a':
        mu_fixed = 3.24
        delta_l =np.log10(73)
    elif alphabet == 'b':
        mu_fixed = 1
        delta_l= 5

k_fixed = - 1  # remember that we put log10 of k in the model
alphaload_fixed = delta_l # * mu_fixed * (np.pi**(2/3))
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 30)  # 100 points between -1 and 6
# n_values = [0.2, 0.4, 0.6, 0.8, 1.6]  # Given n values
n_values = [0.2, 0.6, 1.6]  # Given n values

# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
# colors = sns.color_palette("deep", n_colors=5)  # 5 distinct colors
# colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33A1']  # Vibrant colors
# colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF']  # Soft pastel colors
# colors = ['#A0522D', '#FFD700', '#8FBC8F', '#4682B4', '#B0E0E6']  # Natural colors
# colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']  # Colorblind-friendly options

# trained_model = models['XGBoost']  # Retrieve trained XGBoost model
trained_model = loaded_models.get('XGBoost')
# trained_model = loaded_models.get('Neural Network')

dataPB = load_pb_data(file_pathPB)

# Plotting the results
fig2 = plt.figure()

filtered_dataPB = dataPB[(dataPB['r_ml'] > -2) & (dataPB['r_ml'] < 14)]
# Extract the columns of interest
r_ml = filtered_dataPB['r_ml']
# n_columns = filtered_dataPB.columns[1:6]  # Extracting columns from n_0.2 to n_1.6
n_columns = filtered_dataPB.columns[[1, 3, 5]]  # Extracting columns with indices 1, 3, and 5

# Plotting the results for r_ml between -2 and 6
for i, n_col in enumerate(n_columns):
    plt.plot( np.power(10, r_ml), np.power(10,filtered_dataPB[n_col]), label=n_col, color=colors[i], lw=2, alpha=0.4, linestyle='--')
# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, n in enumerate(n_values):
    # Use plt.plot() to create a dashed line instead of plt.scatter()
    # plt.scatter(runload_values, predictions[n], label=f'n = {n}', marker='*', s=100, color=colors[i])
    plt.scatter(10** runload_values, 10** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.4,
             lw=5)
    # plt.plot(runload_values, predictions[n], linestyle='--', label=f'n = {n}', lw=2)



# plt.title('Normalized surface energy vs Unloading rate for Different Values of n', fontsize=16)


# Show plot


# Add legend
# plt.legend(title='n values')

# Save the figure
# plt.tight_layout()

######################################################
# Load the CSV file into a DataFrame
dataML = pd.read_csv(file_path_RPML_5inputs)

# Call the utility function to process the columns.
# It means we are making some changes in dimensions, please check utilities.py file
dataML = process_columns(dataML)

filtered_data = dataML[(dataML['mu'] == mu_fixed) &
                       (dataML['k'] == k_fixed) &
                       (dataML['alphaload'] > alphaload_fixed) &
                       (dataML['alphaload'] < alphaload_fixed + 1)
]

# Check if filtered_data is not empty before plotting
if not filtered_data.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, n_value in enumerate(n_values):
        # Filter data for the current n_value
        data_n = filtered_data[filtered_data['n'] == n_value]
        data_n_sorted = data_n.sort_values(by=['runload'])  # Optional sorting

        # Drop duplicates based on 'runload', keeping only the first occurrence
        data_n_unique = data_n_sorted.drop_duplicates(subset=['runload'], keep='first')

        # Plot outputAmplification vs. runload for the current n_value
        #plt.scatter(np.power(10,data_n['runload']), np.power(10, data_n['outputAmplification']), label=f'n = {n_value}', color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.9, s=70)
        plt.scatter(10** data_n_unique['runload'], 10**data_n_unique['outputAmplification'], label=f'n = {n_value}', color='black', alpha=0.9, edgecolor='black', linewidth=0.9, s=50, marker='D')

        # plt.scatter(data_n['runload'], data_n['outputAmplification'], label=f'n = {n_value}', color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.9, s=70)


        # plt.scatter(data_n['runload'], data_n['outputAmplification'],
         #       color=colors[i], alpha=0.3, edgecolor='none', s=70)  # Core with alpha 0.3

    # Add plot labels and title
    # plt.xlabel('Runload')
    # plt.ylabel('Output Amplification')
    # plt.title('Output Amplification vs Runload for Different Values of n')
    # plt.legend(title='n values')
    # plt.grid(True)

    # Show the plot



else:
    print("No data found in the filtered dataset.")

# Add a text arrow
plt.annotate('$n= [0.2, 0.6, 1.6]$', xytext=(10 ** -1.95, 10 ** 0.9), xy=(10 ** 3, 10 ** 0.3),
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             fontsize=font_size)
plt.text(10**6, 10**0.3, f'$\\mu={mu_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(10**6, 10**0.15, label, fontsize=font_size)
plt.text(10**8.05, 10**0.001, f'({alphabet})', fontsize=font_size)

# Adjust x and y axis limits
plt.xlim(10**-2, 10**9)  # Set x-axis limits from 0 to 5

plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{\Gamma}_{eff}$')
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
filename = f"RPML_effective_surf_ener_vs_r_mu_{mu_fixed}_d_73.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()
