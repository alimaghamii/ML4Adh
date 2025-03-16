import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from cpsme.export_figure import export_figure
import joblib
import os
# Here, we import details from utilities
from utilities import load_pb_data
# from utilities import file_path_RPML_5inputs_W
from utilities import file_pathPB
from utilities import file_PB
from utilities import save_dir_PARPML_6inputs_W
from utilities import process_columns
from utilities import aug_data_provider
from utilities import file_path_RPML_5inputs_test

save_dir = save_dir_PARPML_6inputs_W
# Ensure the directory exists (optional, if you want to create it if it doesn't exist)
os.makedirs(save_dir, exist_ok=True)

font_size =20
# Set default font to Times New Roman globally, including math font
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

# trained_model = models['XGBoost']  # Retrieve trained XGBoost model
trained_model = loaded_models.get('XGBoost')
# trained_model = loaded_models.get('Neural Network')

dataPB = pd.read_csv(file_pathPB)

# Plotting the results
fig2 = plt.figure()

filtered_dataPB = dataPB[(dataPB['r_ml'] > -2) & (dataPB['r_ml'] < 14)]
# Extract the columns of interest
r_ml = filtered_dataPB['r_ml']
n_columns = filtered_dataPB.columns[[1, 3, 5]]  # Extracting columns with indices 1, 3, and 5

# Plotting the results for r_ml between -2 and 6
def make_predictions_Phys_A(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed):
    predictions = {}
    for n in n_values:
        # Create a DataFrame for input variables
        input_data = pd.DataFrame({
            'runload': runload_values,
            'alphaload': alphaload_fixed,
            'mu': mu_fixed,
            'k': k_fixed,
            'n': n

        })

        data_aug = aug_data_provider(input_data,file_PB)
        # Predict using the trained model
        y_pred = trained_model.predict(data_aug)

        # Store the predictions for each value of n
        predictions[n] = y_pred

    return predictions
########################################################################################################################
def make_predictions_Phys_A2(trained_model, runload_values, n_fixed, mu_values, k_fixed, alphaload_fixed):
    predictions = {}
    for mu in mu_values:
        # Create a DataFrame for input variables
        input_data = pd.DataFrame({
            'runload': runload_values,
            'alphaload': alphaload_fixed,
            'mu': mu,
            'k': k_fixed,
            'n': n_fixed

        })

        data_aug = aug_data_provider(input_data,file_PB)
        # Predict using the trained model
        y_pred = trained_model.predict(data_aug)

        # Store the predictions for each value of n
        predictions[mu] = y_pred

    return predictions
#########################################################################################################################
# Set fixed values for mu and k for fig 8, 9 and 10
figure= 11
alphabet= 'b' # it will be written on the figure

if figure== 11:
    mu_fixed = 3.24
    delta_load= 73
    delta_l= np.log10(delta_load)




k_fixed = - 1  # remember that we put log10 of k in the model
alphaload_fixed = delta_l # * mu_fixed * (np.pi**(2/3))
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 35)  # 100 points between -1 and 6
n_values = [0.4, 0.6, 1.6]  # Given n values
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options


# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, n in enumerate(n_values):
    plt.scatter(10** runload_values, 10** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.4,
             lw=2, s= 90)
    plt.plot(10** runload_values, 10** predictions[n], linestyle='-', color=colors[i], label=f'n = {n}', lw=6, alpha= 0.1)


######################################################
# Load the CSV file into a DataFrame
# dataML = pd.read_csv(file_path_RPML_5inputs_W)
data_test_final = pd.read_csv(file_path_RPML_5inputs_test)

# Call the utility function to process the columns.
# It means we are making some changes in dimensions, please check utilities.py file


data_test_final = process_columns(data_test_final)

filtered_data_test_final = data_test_final[(data_test_final['mu'] == mu_fixed) &
                       (data_test_final['k'] == k_fixed) &
                       (data_test_final['alphaload'] > alphaload_fixed) &
                       (data_test_final['alphaload'] < alphaload_fixed + 0.2)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, n_value in enumerate(n_values):
        # Filter data for the current n_value
        data_n = filtered_data_test_final[filtered_data_test_final['n'] == n_value]

        plt.scatter(10** data_n['runload'], 10**data_n['workofSepration'], label=f'n = {n_value}', color=colors2[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')

else:
    print("No data found in the filtered dataset.")

# Add a text arrow
plt.annotate('$n= [0.4, 0.6, 1.6]$', xytext=(10 ** 0, 10 ** 2.25), xy=(10 ** 2, 26),
              arrowprops=dict(facecolor='black', arrowstyle='<-'),
              fontsize=font_size)
plt.text(10**4, 100, f'$\\mu={mu_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(10**4, 65, label, fontsize=font_size)
plt.text(10**5.5, 16, f'({alphabet})', fontsize=font_size)
#
# Adjust x and y axis limits
plt.xlim(1, 10**6)  # Set x-axis limits from 0 to 5
plt.ylim(12, 155)  # Set x-axis limits from 0 to 5

####################
x_range = np.logspace(2, 6, 100)  # 100 points between 10^4 and 10^6
w_po = 1.1818 * 3.2 * np.pi**(2/3) * (10)**(2/3)
plt.plot(x_range, [w_po] * len(x_range), lw=3,alpha= 0.4, linestyle='--', color='black')

####################


plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{w}_{po}$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
filename = f"PARPML_W_vs_r_mu_{mu_fixed}_d_{delta_load}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()
#################################################### mu effect figures #################################################
# Set fixed values for n and k for mu study
#########################################################################################################################
# Set fixed values for mu and k for fig 8, 9 and 10
figure= 11
alphabet= 'a' # it will be written on the figure

if alphabet == 'a':
    n_fixed = 1.6
    delta_load=73
    delta_l= np.log10(delta_load)
    delta_load
elif alphabet == 'b':
    n_fixed = 0.6
    delta_load=73
    delta_l = np.log10(delta_load)

k_fixed = - 1  # remember that we put log10 of k in the model
alphaload_fixed = delta_l # * mu_fixed * (np.pi**(2/3))
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 30)  # 100 points between -1 and 6
mu_values = [0.05, 0.2, 0.8, 3.2]  # Given n values
# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A2(trained_model, runload_values, n_fixed, mu_values, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, mu in enumerate(mu_values):
    # Use plt.plot() to create a dashed line instead of plt.scatter()
    plt.scatter(10 ** runload_values, 10 ** predictions[mu], label=f'mu = {mu}', color=colors[i], alpha=0.4,
                lw=2, s=90)
    plt.plot(10 ** runload_values, 10 ** predictions[mu], linestyle='-', color=colors[i], label=f'mu = {mu}', lw=6,
             alpha=0.1)

######################################################
# Load the CSV file into a DataFrame
data_test_final = pd.read_csv(file_path_RPML_5inputs_test)

data_test_final = process_columns(data_test_final)

filtered_data_test_final = data_test_final[(data_test_final['n'] == n_fixed) &
                       (data_test_final['k'] == k_fixed) &
                       (data_test_final['alphaload'] > alphaload_fixed) &
                       (data_test_final['alphaload'] < alphaload_fixed + 0.2)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data_test_final[filtered_data_test_final['n'] == mu_value]

        plt.scatter(10** data_mu['runload'], 10**data_mu['workofSepration'], label=f'mu = {mu_value}', color=colors2[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')

else:
    print("No data found in the filtered dataset.")

##############################
# Define x-axis range
x_range = np.logspace(3, 6, 100)  # 100 points between 10^4 and 10^6

# Calculate w_po for each mu
# mu = 3.2
w_po = 1.1818 * 3.2 * np.pi**(2/3) * (10)**(2/3)

# Create the plot
plt.plot(x_range, [w_po] * len(x_range),lw=3,linestyle= '--',
         label=f'mu = {mu}', color=(0.4, 0.0, 0.4))
plt.text(5950, w_po + 15, 'JKR-glassy limit', fontsize=font_size, color=(0.3, 0.0, 0.3))


# Add a text arrow
plt.annotate('$\\mu= [0.05, 0.2, 0.8, 3.2]$', xytext=(10 ** 0, 10 ** 2.29), xy=(500, 3),
              arrowprops=dict(facecolor='black', arrowstyle='<-'),
              fontsize=font_size)
plt.text(850, 1.75, f'$n={n_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(850, 0.9, label, fontsize=font_size)
plt.text(10**5.5, 1.1, f'({alphabet})', fontsize=font_size)
#
# Adjust x and y axis limits
plt.xlim(1, 10**6)  # Set x-axis limits from 0 to 5
plt.ylim(0.7, 165)  # Set x-axis limits from 0 to 5

plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{w}_{po}$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
filename = f"PARPML_W_vs_r_n_{n_fixed}_d_{delta_load}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()