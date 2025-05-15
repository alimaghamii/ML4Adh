import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from cpsme.export_figure import export_figure
import joblib
import os
from matplotlib.patches import ConnectionPatch, Rectangle

# Here, we import details from utilities
from utilities import load_pb_data
from utilities import file_path_RPML_5inputs
from utilities import file_pathPB
from utilities import file_PB
from utilities import save_dir_PARPML_6inputs
from utilities import process_columns
from utilities import aug_data_provider
from utilities import file_path_RPML_5inputs_test

save_dir = save_dir_PARPML_6inputs
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

# Set fixed values for mu and k
figure = 8 # specify the figure that you want to creat
# 6, 7, or 8
alphabet = 'd'

if figure== 6: 
    alphabet = 'b'
    mu_fixed =3.24
    delta_l0 = 73
    delta_l =np.log10(delta_l0)

elif figure== 7:
    if alphabet=='a':
        mu_fixed = 3.0
        delta_l0 = 60
        delta_l = np.log10(delta_l0)
    elif alphabet == 'b':
        mu_fixed = 3.0
        delta_l0 = 30
        delta_l = np.log10(delta_l0)
    elif alphabet == 'c':
        mu_fixed = 3.0
        delta_l0 = 15
        delta_l = np.log10(delta_l0)
    elif alphabet == 'd':
        mu_fixed = 3.0
        delta_l0 = 7.5
        delta_l = np.log10(delta_l0)

elif figure== 8:
    if alphabet=='a':
        mu_fixed = 2.56
        delta_l0 = 75
        delta_l = np.log10(delta_l0)
    elif alphabet == 'b':
        mu_fixed = 0.16
        delta_l0 = 75
        delta_l= np.log10(delta_l0)
    elif alphabet=='c':
        mu_fixed = 0.16
        delta_l0 = 15
        delta_l = np.log10(delta_l0)
    elif alphabet == 'd':
        mu_fixed = 0.16
        delta_l0 = 7.5
        delta_l = np.log10(delta_l0)

k_fixed = - 1  # remember that we put log10 of k in the model
# alphaload_fixed = delta_l * mu_fixed * (np.pi**(2/3))
alphaload_fixed = delta_l
# Define the range of runload (unloading rate) and n values
runload_values = np.linspace(-1.5, 10, 30)  # 100 points between -1 and 6
# n_values = [0.2, 0.4, 0.6, 0.8, 1.6]  # Given n values
n_values = [0.2, 0.6, 1.6]  # Given n values
# n_values = np.log10(n_values)
colors = ['red', 'blue', 'green', 'purple', '#E69F00']  # Define the colors
colors2 = ['#B00050', '#00008B', '#006400']
colors2 = ['#D10070', '#0033CC', '#008F11']
colors2 = ['#D10070', '#007FFF', '#009970']

# trained_model = models['XGBoost']  # Retrieve trained XGBoost model
# trained_model = loaded_models.get('XGBoost')
trained_model = loaded_models.get('Random Forest')
# trained_model = loaded_models.get('Neural Network')

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


for i, n_col in enumerate(n_columns):
    plt.plot(10** r_ml, 10** filtered_dataPB[n_col], label=n_col, color=colors[i], lw=2, alpha=0.4, linestyle='--')



# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed)

# Plot predictions for each value of n
for i, n in enumerate(n_values):
    # plt.plot(10** runload_values, 10** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.3,
    #         lw=6)
    plt.scatter(10** runload_values, 10** predictions[n], label=f'n = {n}', color=colors[i], alpha=0.4,
             lw=5)
######################################################
# Load the CSV file into a DataFrame
dataML = pd.read_csv(file_path_RPML_5inputs)
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
        # Ensure data is sorted in the order you want to keep the first occurrence
        data_n_sorted = data_n.sort_values(by=['runload'])  # Optional sorting

        # Drop duplicates based on 'runload', keeping only the first occurrence
        data_n_unique = data_n_sorted.drop_duplicates(subset=['runload'], keep='first')

        plt.scatter(10** data_n_unique['runload'], 10**data_n_unique['outputAmplification'], label=f'n = {n_value}', color='black', alpha=0.9, edgecolor='black', linewidth=0.9, s=50, marker='D')

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

        plt.scatter(10** data_n['runload'], 10**data_n['outputAmplification'], label=f'n = {n_value}', color=colors2[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')

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
plt.xlim(10 ** -2, 10 ** 9)  # Set x-axis limits from 0 to 5

# export_figure(fig2, name='RPML_effective_surf_ener_vs_r.png', style='presentation_1x1', savedir= save_dir)
# Apply tight_layout to make sure labels are not cut off
# Set both x and y axes to logarithmic scale
# Add labels and title
plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{\Gamma}_{eff}$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
filename = f"PARPML_effective_surf_ener_vs_r_mu_{mu_fixed}_d_{delta_l0}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()


#######################################################################################################################
#################################################### mu effect figures #################################################
# Set fixed values for runload and k for mu study
font_size= 25
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
    'figure.figsize': (15, 5)          # Default figure size
})
fig = plt.figure(figsize=(15, 5))

# Manually place two Axes objects side by side
# left, bottom, width, height
ax_main = fig.add_axes([0.07, 0.23, 0.5, 0.765])   # Main plot on the left
ax_zoom = fig.add_axes([0.6, 0.3, 0.38, 0.5])  # Zoomed plot on the right

# mu_values = [0.04, 0.1, 0.15, 0.3, 0.5, 1, 2, 3.24]
mu_values = [0.2, 0.8, 3.2]

# colors = ['red', 'blue', 'green', 'purple', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF', '#32CD32',
#     '#FFD700', '#8A2BE2', '#FF1493', '#00FF00']

# colors = ['red', 'blue', 'green', 'darkorange', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF']
colors = ['red',  'blue', 'green', 'red', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF']

# mu_values = [0.04, 0.5, 1]

#alphaload_values= np.logspace(0.1, 2, 7)
alphaload_values= np.log10 ([1 ,4 , 8, 16, 32, 64])
logvalues = np.logspace(0, np.log10(64), 20)
alphaload_values= np.log10 (logvalues)


# alphaload_values= np.log10 ([1,2 ,4,6 ,8, 12, 16, 24, 32, 48, 64, 75])
# alphaload_values= np.linspace(1, 100, 7)

k_fixed = -1 # remember that we put log10 of k in the model
runload_fixed = 8

n_fixed = 0.6  # Given n values

def make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values):
    predictions = {}
    for mu in mu_values:
        # Create a DataFrame for input variables
        input_data = pd.DataFrame({
            'runload': runload_fixed,
            'alphaload': alphaload_values,
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

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values)



# Plot predictions for each value of n
for i, mu in enumerate(mu_values):
   # ax_main.plot(10** alphaload_values, 10** predictions[mu],color=colors[i], label=f'n = {n}', alpha=0.4,
   #          lw=6)

    ax_main.scatter(10** alphaload_values, 10** predictions[mu], label=f'n = {n}', color=colors[i], alpha=0.6,
             lw=8)

filtered_data2 = dataML[(dataML['k'] == k_fixed) &
                       (dataML['runload'] > runload_fixed-0.001) &
                       (dataML['runload'] < runload_fixed+ 0.01) &
                        (dataML['n'] == n_fixed)
]

# Check if filtered_data is not empty before plotting
if not filtered_data2.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data2[filtered_data2['mu'] == mu_value]

        ax_main.scatter(10** data_mu['alphaload'], 10** data_mu['outputAmplification'], label=f'mu = {mu_value}', color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.9, s=70)

else:
    print("No data found in the filtered dataset.")

ax_main.plot(10** alphaload_values, [10] * len(alphaload_values), color='dimgray', lw=4, alpha=0.7, linestyle='--')
ax_main.text(1.1, 10.3, 'XPB', fontsize=font_size, color='black')  # Add text at the top-left corner of the ax_main figure

#ax_main.xscale('log')
ax_main.set_xscale('log')
ax_main.set_yscale('log')
#ax_main.yscale('log')
# ax_main.xlabel(r'$\widehat{\delta}_l$')
ax_main.set_xlabel(r'$\widehat{\delta}_l$')
ax_main.set_ylabel(r'$\widehat{\Gamma}_{eff}$')
#ax_main.ylabel(r'$\widehat{\Gamma}_{eff}$')
# ax_main.xlim([1,75])
ax_main.set_xlim([1, 64])  # <-- Your desired x-limits for the main figure
#ax_main.yticks([2, 3, 4, 6, 10], labels=[" "," "," "," ","10"])  # Only show the tick at y=10
# Add a text arrow
ax_main.set_yticks([2, 3, 4, 6, 10])
ax_main.set_yticklabels([" "," "," "," ","10"])

ax_main.set_xticks([1, 10, 64])
ax_main.set_xticklabels(["1","10","64"])

filtered_data_test_final = data_test_final[(data_test_final['k'] == k_fixed) &
                       (data_test_final['runload'] > runload_fixed-0.001) &
                       (data_test_final['runload'] < runload_fixed+ 0.01)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data_test_final[filtered_data_test_final['mu'] == mu_value]

        ax_main.scatter(10** data_mu['alphaload'], 10**data_mu['outputAmplification'], label=f'n = {mu_value}', color=colors[i],edgecolor='black', alpha=1, linewidth=1, s=176, marker='^')

else:
    print("No data found in the filtered dataset.")


ax_main.annotate('$\\mu= [0.2, 0.8, 3.2]$',
             xy=(5, 3),          # start point of the arrow (x, y)
             xytext=(1.1, 6.5),      # text point (x, y)
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             fontsize=font_size)
ax_main.text(10**1.4, 2, f'$n={n_fixed}$', fontsize=font_size, color='black')
lable = label= f'$\\widehat{{r}}_u=10^{runload_fixed}$'
ax_main.text(10**1.4, 2.5, label, fontsize=font_size)
alphabet= 'a'
# ax_main.text(54, 2, f'({alphabet})', fontsize=font_size)
fig.text(0.74, 0.1, f'({alphabet})', fontsize=font_size)

# -------------------------------------------------------------
# 4. Identify region to "zoom" in on, and configure the zoomed axes
# -------------------------------------------------------------
# For demonstration, let's zoom in on x-range [2, 4].
x_min, x_max = 15, 64
y_min, y_max = 7, 18

# Plot the same curve on the zoomed-in axes, focusing on the narrower range
# ax_zoom.plot(x, y, color='blue', linewidth=2)
for i, mu in enumerate(mu_values):
    # ax_zoom.plot(10** alphaload_values, 10** predictions[mu],color=colors[i], label=f'n = {n}', alpha=0.4,
    #          lw=6)

    ax_zoom.scatter(10** alphaload_values, 10** predictions[mu], label=f'n = {n}', color=colors[i], alpha=0.6,
             lw=8)
    #data_mu = filtered_data_test_final[filtered_data_test_final['mu'] == mu_value]

    # ax_zoom.scatter(10** data_mu['alphaload'], 10**data_mu['outputAmplification'], label=f'n = {mu_value}', color=colors[i],edgecolor='black', alpha=1, linewidth=1, s=120, marker='^')


filtered_data_test_final = data_test_final[(data_test_final['k'] == k_fixed) &
                       (data_test_final['runload'] > runload_fixed-0.001) &
                       (data_test_final['runload'] < runload_fixed+ 0.01)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data_test_final[filtered_data_test_final['mu'] == mu_value]

        ax_zoom.scatter(10** data_mu['alphaload'], 10**data_mu['outputAmplification'], label=f'n = {mu_value}', color=colors[i],edgecolor='black', alpha=1, linewidth=1, s=176, marker='^')

else:
    print("No data found in the filtered dataset.")



ax_zoom.set_xlim([x_min, x_max])
ax_zoom.set_ylim([y_min, y_max])
ax_zoom.set_xscale('log')
ax_zoom.set_yscale('log')
#ax_main.yscale('log')
# ax_zoom.set_title('Zoomed-In View')
ax_zoom.set_yticks([7, 8, 9, 10, 40/3])
ax_zoom.set_yticklabels([
    " ",
    " ",
    " ",
    r'$\frac{1}{k}$',
    r'$\frac{4}{3k}$'
], fontsize=32)
ax_zoom.set_xticks([15 ,20 , 30, 40, 60, 64])
ax_zoom.set_xticklabels([15, "","","","", 64])

ax_zoom.text(15, 10.1, 'JKR-like limit', fontsize=22)
ax_zoom.text(15, 40/3+0.1, 'DMT-like limit', fontsize=22)

# ax_zoom.plot(alphaload_values, [10] * len(alphaload_values), color='dimgray',  lw=4, alpha=0.7, linestyle='--')
ax_zoom.plot(10** alphaload_values, [10] * len(alphaload_values), color='brown',  lw=3, alpha=0.5, linestyle='-.')
ax_zoom.plot(10** alphaload_values, [20/1.5] * len(alphaload_values),color='black', lw=3, alpha=1.0, linestyle='-.')
# Example of custom ticks for the zoomed-in (linear) axis
#ax_zoom.set_xticks([2, 3, 4])
#ax_zoom.set_xticklabels(['2', '3', '4'])
ax_zoom.annotate('$\\mu= [0.2, 0.8, 3.2]$',
             xy=(52, 8.5),          # start point of the arrow (x, y)
             xytext=(22, 15),      # text point (x, y)
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=font_size)
# -------------------------------------------------------------
# 5. Draw "projection" lines connecting the main plot to the zoomed plot
# -------------------------------------------------------------
# We'll connect the corners of the zoom region in ax_main to the edges of ax_zoom.
# In a log-scaled main plot, we must be careful to use the correct data transform.

# Corner points in data coordinates for the main plot
corners_main = [
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_min),
    (x_max, y_max),
]

# The corresponding corners in the zoomed plot
corners_zoom = [
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_min),
    (x_max, y_max),
]

# Create a dashed line between each pair of corners
for (xm, ym), (xz, yz) in zip(corners_main, corners_zoom):
    # ConnectionPatch from ax_main to ax_zoom
    con = ConnectionPatch(
        xyA=(xm, ym), coordsA=ax_main.transData,  # point in main-axes coords
        xyB=(xz, yz), coordsB=ax_zoom.transData,  # point in zoom-axes coords
        color='gray', linewidth=1, linestyle='--',
        alpha=0.5  # set the transparency
    )
    fig.add_artist(con)

# -------------------------------------------------------------
# 6. Add a rectangle on the main plot around the zoom region
# -------------------------------------------------------------
# This rectangle is drawn in the main plot, highlighting the region [x_min, x_max] × [y_min, y_max].
rect_width  = x_max - x_min
rect_height = y_max - y_min

rect = Rectangle((x_min, y_min), rect_width, rect_height,
                 fill=False, edgecolor='black', linewidth=1, alpha=0.4,
                 transform=ax_main.transData)
ax_main.add_patch(rect)

# -------------------------------------------------------------
# 7. Final adjustments and display
# -------------------------------------------------------------
filename = f"PARPML_effective_surf_ener_vs_delta_n_{n_fixed}_r_{runload_fixed}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()
######################################################################################################################
#################################################### mu effect figures #################################################
# Set fixed values for runload and k for mu study
# font_size=24
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
    'figure.figsize': (9, 5)          # Default figure size
})


# mu_values = [0.04, 0.1, 0.15, 0.3, 0.5, 1, 2, 3.24]
mu_values = [0.2, 0.8, 3.2]

# colors = ['red', 'blue', 'green', 'purple', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF', '#32CD32',
#     '#FFD700', '#8A2BE2', '#FF1493', '#00FF00']

# colors = ['purple',  'blue', 'green', 'red', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF']
# mu_values = [0.04, 0.5, 1]

alphaload_values= np.linspace(1, 100, 7)
alphaload_values= np.log10 ([1, 4, 8, 16, 32, 64, 100])
logvalues = np.logspace(0, np.log10(64), 20)
alphaload_values= np.log10 (logvalues)


k_fixed = -1 # remember that we put log10 of k in the model
runload_fixed = 3

n_fixed = 0.6  # Given n values

def make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values):
    predictions = {}
    for mu in mu_values:
        # Create a DataFrame for input variables
        input_data = pd.DataFrame({
            'runload': runload_fixed,
            'alphaload': alphaload_values,
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

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values)


# Plot predictions for each value of mu
for i, mu in enumerate(mu_values):
    # plt.plot(10** alphaload_values, 10** predictions[mu],color=colors[i], label=f'n = {n}', alpha=0.4,
    #          lw=6)
    plt.scatter(10** alphaload_values, 10** predictions[mu], label=f'n = {n}', color=colors[i], alpha=0.6,
             lw=8)

filtered_data2 = dataML[(dataML['k'] == k_fixed) &
                       (dataML['runload'] > runload_fixed-0.001) &
                       (dataML['runload'] < runload_fixed+ 0.01)
]

plt.plot(10** alphaload_values, [8] * len(alphaload_values), color='dimgray',  lw=4, alpha=0.8, linestyle='--')
plt.text(1.1, 8.3, 'XPB', fontsize=font_size, color='black')  # Add text at the top-left corner of the figure

# plot(alphaload_values, [10] * len(alphaload_values), color='brown',  lw=2, alpha=0.5, linestyle='-.')
# plt.plot(alphaload_values, [20/1.5] * len(alphaload_values),color='black', lw=2, alpha=1.0, linestyle='-.')
# note: think about the scale of outputamplification.
# Set the y-axis tick to only show y = 10
# plt.yticks([10], labels=["10"])  # Only the desired value will have a label
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\widehat{\delta}_l$')
plt.ylabel(r'$\widehat{\Gamma}_{eff}$')
plt.xlim([1,64])
plt.yticks([2, 3, 4, 6, 10], labels=[" "," "," "," ","10"])  # Only show the tick at y=10
# Add a text arrow
# plt.xticks([2, 3, 4, 6, 10], labels=[" "," "," "," ","10"])  # Only show the tick at y=10
plt.ylim([1,10])
# The desired tick locations
ticks = [1, 10, 64]
labels = [str(t) for t in ticks]
plt.xticks(ticks, labels)


filtered_data_test_final = data_test_final[(data_test_final['k'] == k_fixed) &
                       (data_test_final['runload'] > runload_fixed-0.001) &
                       (data_test_final['runload'] < runload_fixed+ 0.01)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data_test_final[filtered_data_test_final['mu'] == mu_value]

        plt.scatter(10** data_mu['alphaload'], 10**data_mu['outputAmplification'], label=f'n = {mu_value}', color=colors[i],edgecolor='black', alpha=1, linewidth=1, s=176, marker='^')

else:
    print("No data found in the filtered dataset.")


plt.annotate('$\\mu= [0.2, 0.8, 3.2]$',
             xy=(5, 3),          # start point of the arrow (x, y)
             xytext=(1.1, 6.5),      # text point (x, y)
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             fontsize=font_size)
plt.text(10**1.4, 2, f'$n={n_fixed}$', fontsize=font_size, color='black')
lable = label= f'$\\widehat{{r}}_u=10^{runload_fixed}$'
plt.text(10**1.4, 2.5, label, fontsize=font_size)
alphabet= 'b'
plt.text(48, 1.2, f'({alphabet})', fontsize=font_size)
plt.tight_layout()
filename = f"PARPML_effective_surf_ener_vs_delta_n_{n_fixed}_r_{runload_fixed}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()

######################################################################################################################
#################################################### mu effect figures #################################################
# Set fixed values for runload and k for mu study

# mu_values = [0.04, 0.1, 0.15, 0.3, 0.5, 1, 2, 3.24]
mu_values = [0.2, 0.8, 3.2]

# colors = ['red', 'blue', 'green', 'purple', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF', '#32CD32',
#     '#FFD700', '#8A2BE2', '#FF1493', '#00FF00']

# colors = ['red', 'blue', 'green', 'darkorange', '#E69F00',  '#FF4500', '#00CED1', '#1E90FF']
# mu_values = [0.04, 0.5, 1]

alphaload_values= np.linspace(1, 1000, 50)
alphaload_values= np.log10 ([1, 4, 8, 16, 32, 64, 100])
logvalues = np.logspace(0, np.log10(64), 20)
alphaload_values= np.log10 (logvalues)


k_fixed = -1 # remember that we put log10 of k in the model
runload_fixed = 1

n_fixed = 0.6  # Given n values

def make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values):
    predictions = {}
    for mu in mu_values:
        # Create a DataFrame for input variables
        input_data = pd.DataFrame({
            'runload': runload_fixed,
            'alphaload': alphaload_values,
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

# Make predictions. It gets the trained model and 5 inputs
predictions = make_predictions_Phys_A2(trained_model, runload_fixed, n_fixed, mu_values, k_fixed, alphaload_values)


# Plot predictions for each value of n
for i, mu in enumerate(mu_values):
    #plt.plot(10** alphaload_values, 10** predictions[mu],color=colors[i], label=f'n = {n}', alpha=0.4,
    #         lw=6)
    plt.scatter(10** alphaload_values, 10** predictions[mu], label=f'n = {n}', color=colors[i], alpha=0.6,
             lw=8)

filtered_data2 = dataML[(dataML['k'] == k_fixed) &
                       (dataML['runload'] > runload_fixed-0.001) &
                       (dataML['runload'] < runload_fixed+ 0.01)
]

plt.plot(10** alphaload_values, [2.4] * len(alphaload_values), color='dimgray',  lw=4, alpha=0.7, linestyle='--')
plt.text(1.1, 2.5, 'XPB', fontsize=font_size, color='black')  # Add text at the top-left corner of the figure

# plt.plot(alphaload_values, [10] * len(alphaload_values), color='brown',  lw=2, alpha=0.5, linestyle='-.')
# plt.plot(alphaload_values, [20/1.5] * len(alphaload_values),color='black', lw=2, alpha=1.0, linestyle='-.')
# note: think about the scale of outputamplification.
# Set the y-axis tick to only show y = 10
# plt.yticks([10], labels=["10"])  # Only the desired value will have a label
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\widehat{\delta}_l$')
plt.ylabel(r'$\widehat{\Gamma}_{eff}$')
plt.xlim([1,64])
plt.ylim([1,10])
plt.yticks([2, 3, 4, 6, 10], labels=[" "," "," "," ","10"])  # Only show the tick at y=10
# Add a text arrow

filtered_data_test_final = data_test_final[(data_test_final['k'] == k_fixed) &
                       (data_test_final['runload'] > runload_fixed-0.001) &
                       (data_test_final['runload'] < runload_fixed+ 0.01)
]

# Check if filtered_data is not empty before plotting
if not filtered_data_test_final.empty:
    # plt.figure(figsize=(10, 6))  # Create a new figure for the plot

    # Loop through unique values of 'n' in the filtered_data
    for i, mu_value in enumerate(mu_values):
        # Filter data for the current n_value
        data_mu = filtered_data_test_final[filtered_data_test_final['mu'] == mu_value]

        plt.scatter(10** data_mu['alphaload'], 10**data_mu['outputAmplification'], label=f'n = {mu_value}', color=colors[i],edgecolor='black', alpha=1, linewidth=1, s=176, marker='^')

else:
    print("No data found in the filtered dataset.")


plt.annotate('$\\mu= [0.2, 0.8, 3.2]$',
             xy=(10, 1.7),          # start point of the arrow (x, y)
             xytext=(3, 3),      # text point (x, y)
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             fontsize=font_size)
plt.text(10**1.4, 4, f'$n={n_fixed}$', fontsize=font_size, color='black')
lable = label= f'$\\widehat{{r}}_u=10^{runload_fixed}$'
plt.text(10**1.4, 4.9, label, fontsize=font_size)
alphabet= 'c'
plt.text(48, 1.2, f'({alphabet})', fontsize=font_size)
plt.ylim([1,10])
# The desired tick locations
ticks = [1, 10, 64]
labels = [str(t) for t in ticks]
plt.xticks(ticks, labels)
plt.tight_layout()
filename = f"PARPML_effective_surf_ener_vs_delta_n_{n_fixed}_r_{runload_fixed}.png"
plt.savefig(f"{save_dir}/{filename}")
plt.show()