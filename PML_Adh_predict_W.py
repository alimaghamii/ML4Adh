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
from utilities import file_path_RPML_5inputs_W
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
trained_model = loaded_models.get('Random Forest')

dataPB = pd.read_csv(file_pathPB)

# Plotting the results
fig2 = plt.figure()

filtered_dataPB = dataPB[(dataPB['r_ml'] > -2) & (dataPB['r_ml'] < 14)]
# Extract the columns of interest
r_ml = filtered_dataPB['r_ml']
# n_columns = filtered_dataPB.columns[1:6]  # Extracting columns from n_0.2 to n_1.6
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

# elif figure ==12:
#     if alphabet == 'a':
#         mu_fixed = 3.0
#         delta_l = np.log10(60)
#     elif alphabet =='b':
#         mu_fixed = 3.0
#         delta_l = np.log10(30)
#     elif alphabet == 'c':
#         mu_fixed = 2.56
#         delta_l = np.log10(75)
#     elif alphabet == 'd':
#         mu_fixed = 0.04
#         delta_l = np.log10(75)


k_fixed = - 1  # remember that we put log10 of k in the model
alphaload_fixed = delta_l # * mu_fixed * (np.pi**(2/3))
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 35)  # 100 points between -1 and 6
# n_values = [0.2, 0.4, 0.6, 0.8, 1.6]  # Given n values
n_values = [0.4, 0.6, 1.6]  # Given n values
# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options

# for i, n_col in enumerate(n_columns):
#     plt.plot(10** r_ml, 10** filtered_dataPB[n_col], label=n_col, color=colors[i], lw=3, alpha=0.3, linestyle='--')

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, n in enumerate(n_values):
    # Use plt.plot() to create a dashed line instead of plt.scatter()
    # plt.scatter(runload_values, predictions[n], label=f'n = {n}', marker='*', s=100, color=colors[i])
    # plt.scatter(10 ** runload_values, 10 ** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.1,
    #             lw=2)
    # plt.plot(10 ** runload_values, 10 ** predictions[n], linestyle='-', color=colors[i], label=f'n = {n}', lw=4,
    #          alpha=0.3)
    plt.scatter(10** runload_values, 10** predictions[n]/1.5, label=f'n = {n}', color=colors[i], alpha=0.4,
             lw=2, s= 90)
    plt.plot(10** runload_values, 10** predictions[n]/1.5, linestyle='-', color=colors[i], label=f'n = {n}', lw=6, alpha= 0.1)


######################################################
# Load the CSV file into a DataFrame
dataML = pd.read_csv(file_path_RPML_5inputs_W)
data_test_final = pd.read_csv(file_path_RPML_5inputs_test)

# Call the utility function to process the columns.
# It means we are making some changes in dimensions, please check utilities.py file
dataML = process_columns(dataML)

filtered_data = dataML[(dataML['mu'] == mu_fixed) &
                       (dataML['k'] == k_fixed) &
                       (dataML['alphaload'] > alphaload_fixed) &
                       (dataML['alphaload'] < alphaload_fixed + 0.1)
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
        # plt.scatter(np.power(10,data_n['runload']), np.power(10, data_n['outputAmplification']), label=f'n = {n_value}', color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.9, s=70)
        plt.scatter(10 ** data_n_unique['runload'], 10 ** data_n_unique['workofSepration']/1.5, label=f'n = {n_value}',
                    color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.9, s=50, marker='D')

else:
    print("No data found in the filtered dataset.")


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

        plt.scatter(10** data_n['runload'], 10**data_n['workofSepration']/1.5, label=f'n = {n_value}', color=colors2[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')

else:
    print("No data found in the filtered dataset.")

# Add a text arrow
plt.annotate('$n= [0.4, 0.6, 1.6]$', xytext=(10 ** 0, 10 ** 2.25/1.5), xy=(10 ** 2, 25/1.5),
              arrowprops=dict(facecolor='black', arrowstyle='<-'),
              fontsize=font_size)
plt.text(10**4, 100/1.5, f'$\\mu={mu_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(10**4, 65/1.5, label, fontsize=font_size)
plt.text(10**5.45, 16/1.5, f'({alphabet})', fontsize=font_size)
#
# Adjust x and y axis limits
plt.xlim(1, 10**6)  # Set x-axis limits from 0 to 5
plt.ylim(12/1.5, 155/1.5)  # Set x-axis limits from 0 to 5

####################
x_range = np.logspace(2, 6, 100)  # 100 points between 10^4 and 10^6
w_po = 1.1818 * 3.2 * np.pi**(2/3) * (10)**(2/3)

# plt.plot(x_range, [w_po] * len(x_range), lw=3,alpha= 0.4, linestyle='--', color='black')

# plt.text(5950, w_po - 5, 'JKR-glassy limit', fontsize=font_size, color=(0.3, 0.0, 0.3))
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
alphabet= 'b' # it will be written on the figure

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
# n_values = [0.2, 0.4, 0.6, 0.8, 1.6]  # Given n values
mu_values = [0.05, 0.2, 0.8, 3.2]  # Given n values
# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A2(trained_model, runload_values, n_fixed, mu_values, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, mu in enumerate(mu_values):
    # Use plt.plot() to create a dashed line instead of plt.scatter()
    # plt.scatter(runload_values, predictions[n], label=f'n = {n}', marker='*', s=100, color=colors[i])
    plt.scatter(10 ** runload_values, 10 ** predictions[mu]/1.5, label=f'mu = {mu}', color=colors[i], alpha=0.4,
                lw=2, s=90)
    plt.plot(10 ** runload_values, 10 ** predictions[mu]/1.5, linestyle='-', color=colors[i], label=f'mu = {mu}', lw=6,
             alpha=0.1)

######################################################
# Load the CSV file into a DataFrame
dataML = pd.read_csv(file_path_RPML_5inputs_W)
data_test_final = pd.read_csv(file_path_RPML_5inputs_test)

# Call the utility function to process the columns.
# It means we are making some changes in dimensions, please check utilities.py file
dataML = process_columns(dataML)

filtered_data = dataML[(dataML['n'] == n_fixed) &
                       (dataML['k'] == k_fixed) &
                       (dataML['alphaload'] > alphaload_fixed) &
                       (dataML['alphaload'] < alphaload_fixed + 0.1)
]

# Check if filtered_data is not empty before plotting
if not filtered_data.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data[filtered_data['mu'] == mu_value]
        data_mu_sorted = data_mu.sort_values(by=['runload'])  # Optional sorting

        # Drop duplicates based on 'runload', keeping only the first occurrence
        data_mu_unique = data_mu_sorted.drop_duplicates(subset=['runload'], keep='first')

        # Plot outputAmplification vs. runload for the current n_value
        # plt.scatter(np.power(10,data_n['runload']), np.power(10, data_n['outputAmplification']), label=f'n = {n_value}', color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.9, s=70)
        plt.scatter(10 ** data_mu_unique['runload'], 10 ** data_mu_unique['workofSepration']/1.5, label=f'mu = {mu_value}',
                    color=colors2[i], alpha=0.9, edgecolor='black', linewidth=0.9, s=50, marker='D')

else:
    print("No data found in the filtered dataset.")


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

        plt.scatter(10** data_mu['runload'], 10**data_mu['workofSepration']/1.5, label=f'mu = {mu_value}', color=colors2[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')

else:
    print("No data found in the filtered dataset.")

##############################
# Define x-axis range
x_range = np.logspace(3, 6, 100)  # 100 points between 10^4 and 10^6

# Calculate w_po for each mu
# w_po_values = [1.5724 * mu * np.pi**(2/3) * (10)**(2/3) for mu in mu_values]
# mu = 3.2
# mu = 3.2
w_po = 1.1818 * 3.2 * np.pi**(2/3) * (10)**(2/3)

# Create the plot
#plt.plot(x_range, [w_po] * len(x_range),lw=3,linestyle= '--',
#         label=f'mu = {mu}', color=(0.4, 0.0, 0.4))
#plt.text(5950, w_po + 15, 'JKR-glassy limit', fontsize=font_size, color=(0.3, 0.0, 0.3))

# mu = 0.05
# w_po = 1.5724 * 0.05 * np.pi**(2/3) * (10)**(2/3)

# Create the plot
# for w_po, mu in zip(w_po_values, mu_values):
# plt.plot(x_range, [w_po] * len(x_range), label=f'mu = {mu}')

    # plt.plot(x_range, [w_po] * len(x_range), label=f'mu = {mu}')
##############################

# Add a text arrow
plt.annotate('$\\mu= [0.05, 0.2, 0.8, 3.2]$', xytext=(10 ** 0, 10 ** 2.29/1.5), xy=(500, 3/1.5),
              arrowprops=dict(facecolor='black', arrowstyle='<-'),
              fontsize=font_size)
plt.text(850, 1.75/1.5, f'$n={n_fixed}$', fontsize=font_size, color='black')
label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(850, 0.9/1.5, label, fontsize=font_size)
plt.text(10**5.45, 1.1/1.5, f'({alphabet})', fontsize=font_size)
#
# Adjust x and y axis limits
plt.xlim(1, 10**6)  # Set x-axis limits from 0 to 5
plt.ylim(0.7/1.5, 165/1.5)  # Set x-axis limits from 0 to 5

plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{w}_{po}$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
filename = f"PARPML_W_vs_r_n_{n_fixed}_d_{delta_load}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()


##################################################################################
# figure 13
# Upper-bound figure
##################################################################################
# Constants
runload_fixed = np.array([5.0])  # log10(runload) = 5
mu_values = [0.05, 0.2, 0.8, 3.2]
delta_load = 73
delta_l = np.log10(delta_load)
k_fixed = -1
alphaload_fixed = delta_l  # assuming this is the same for both n values

# Eq. (E.8) values from the table
wpo_up_16 = [5.3871, 5.3871, 8.7574, 37.1203]

# Storage for ML predictions
wpo_ml_16 = []

# Predict for n = 0.6
# n_fixed = 0.6
# pred_06 = make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_fixed)
# for mu in mu_values:
#    wpo_ml_06.append(10 ** pred_06[mu][0] / 1.5)

# Predict for n = 1.6
n_fixed = 1.6
pred_16 = make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_fixed)
for mu in mu_values:
    wpo_ml_16.append(10 ** pred_16[mu][0] / 1.5)

# Plot for n = 0.6
# plt.plot(mu_values, wpo_ml_06, marker='o', markersize=10, label='ML Prediction', lw=2)
# plt.plot(mu_values, wpo_up_06, marker='s', markersize=10, label='Eq. (E.8)', lw=2)
# plt.xlabel(r'$\mu$')
# plt.ylabel(r'$\widehat{w}_{{po}}$')
# plt.yscale('log')
# plt.grid(True, linestyle='--', alpha=0.6)
# Add annotations
# plt.text(0.052, 32, r'$n=0.6$', fontsize=font_size)
# plt.text(0.052, 22, r'$\widehat{r}_u=10^5$', fontsize=font_size)
# plt.text(3.15, 2.0, r'(a)', fontsize=font_size)

# plt.tight_layout()
# plt.savefig(f"{save_dir}/Fig13_n_0.6.png")
# plt.show()

# Plot for n = 1.6
plt.plot(mu_values, wpo_ml_16, marker='o', markersize=10, label='ML Prediction', lw=2)
plt.plot(mu_values, wpo_up_16, marker='s', markersize=10, label='Eq. (E.8)', lw=2)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\widehat{w}_{{po}}$')
plt.yscale('log')
# plt.grid(True, linestyle='--', alpha=0.6)
# Add annotations
plt.text(0.052, 32, r'$n=1.6$', fontsize=font_size)
plt.text(0.052, 22, r'$\widehat{r}_u=10^5$', fontsize=font_size)
# plt.text(3.14, 1.8, r'(b)', fontsize=font_size)

plt.tight_layout()
plt.savefig(f"{save_dir}/Fig13_n_1.6.png")
plt.show()

######################################################
#figure 14
####################################################
# Fixed settings
n_fixed = 1.6
mu_fixed = 3.2
delta_load = 73
delta_l = np.log10(delta_load)
k_fixed = -1
alphaload_fixed = delta_l

# Unloading rate range (log space)
runload_values = np.linspace(-1.5, 10, 30)
r_u = 10 ** runload_values

# ML prediction for only mu = 3.2
predictions = make_predictions_Phys_A2(trained_model, runload_values, n_fixed, [mu_fixed], k_fixed, alphaload_fixed)
w_po_predicted = 10 ** predictions[mu_fixed] / 1.5

# Upper and lower bounds
wpo_upper = 37.1203
wpo_lower = 5.50376

# Plot
#plt.figure(figsize=(7, 5))

# ML prediction curve
plt.scatter(r_u, w_po_predicted, label=r'ML Prediction', color='blue', alpha=0.4, lw=2, s=90)
plt.plot(r_u, w_po_predicted, linestyle='-', color='blue', lw=6, alpha=0.1)

# Lower bound line and label
plt.hlines(wpo_lower, r_u[0], 10**3, colors='black', linestyles='--', lw=2)
plt.text(1.1, wpo_lower * 1.1, 'JKR-like rubbery lower bound', color='black', fontsize=18)

# Upper bound line and label
plt.hlines(wpo_upper, 10**3, r_u[-1], colors='black', linestyles='--', lw=2)
plt.text(10**3.1, wpo_upper * 1.1, 'Glassy upper bound', color='black', fontsize=18)

# Axes settings
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10**6)
plt.ylim(1, 165/1.5)

# Axis labels
plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{w}_{\mathrm{po}}$')

# Bottom annotation with parameters
plt.text(10, 1.9, r'$n=1.6\qquad \mu=3.2\qquad \widehat{\delta}_l=73.0$', fontsize=font_size)

# Final layout
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_Mu_3.2_with_bounds.png")
plt.show()
