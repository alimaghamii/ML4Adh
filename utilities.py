import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import fsolve
import networkx as nx

#######################################################################################################################
# Create a directory to save the models
save_dir_RPML_5inputs= 'ML_saved'
save_dir_PARPML_6inputs= 'PML_saved'

save_dir_RPML_5inputs_W= 'ML_saved_W'
save_dir_PARPML_6inputs_W= 'PML_saved_W'

# Load the CSV file into a DataFrame (input data)
file_path_RPML_5inputs = 'Data_Files/comb_1to9.csv'  # Replace with the path to your CSV file
file_path_PARPML_6inputs= 'Data_Files/aug_comb_1to9.csv'  # This data is obtained through Data_augmentation.py in Data_study folder
file_path_RPML_5inputs_test = 'Data_Files/fig_8to11_test.csv'

file_path_RPML_5inputs_W = 'Data_Files/comb_1to9_W.csv'  # Replace with the path to your CSV file
file_path_PARPML_6inputs_W= 'Data_Files/aug_comb_1to9_W.csv'  # This data is obtained through Data_augmentation.py in Data_study folder

# Load the PB data
file_pathPB = 'Data_Files/PB_vs_r_ver2.csv'
file_PB='Data_Files/PB_data.csv'
#######################################################################################################################
# LaTeX Table Template
latex_table_template = r"""
    \begin{{table}}[h!]
        \caption{{MSE, R², variance values across different folds, along with average performance and trained model results.}}
        \centering
        \small
        \begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|c|}}
            \hline
            & \textbf{{Fold}} & 1 & 2 & 3 & 4 & 5 & \textbf{{Average}} & \textbf{{Variance}} & \textbf{{Trained}} \\
            \hline
            \multirow{{2}}{{*}}{{{model_name}}} & {{MSE}} & {mse_f1:.4f} & {mse_f2:.4f} & {mse_f3:.4f} & {mse_f4:.4f} & {mse_f5:.4f} & {mse_avg:.4f} & {mse_var:.4f} & {mse_test:.4f} \\
            & {{$R^2$}} & {r2_f1:.4f} & {r2_f2:.4f} & {r2_f3:.4f} & {r2_f4:.4f} & {r2_f5:.4f} & {r2_avg:.4f} & {r2_var:.4f} & {r2_test:.4f} \\
            \hline
        \end{{tabular}}
        \label{{tab:mse_r2}}
    \end{{table}}
    """
#######################################################################################################################
# Function to process the columns in the DataFrame
def process_columns(data):
    # Check and process 'runload' column if present
    if 'runload' in data.columns and 'mu' in data.columns:
        data['runload'] = np.log10(np.abs(data['runload']))

    if 'alphaload' in data.columns and 'mu' in data.columns:
       data['alphaload'] = np.log10(data['alphaload']) 

    # Check and process 'outputAmplification' column if present
    if 'outputAmplification' in data.columns:
        data['outputAmplification'] = np.log10(np.abs(data['outputAmplification']))
    if 'workofSepration' in data.columns:
        data['workofSepration'] = np.log10(np.abs(data['workofSepration']))
    # Check and process 'k' column if present
    if 'k' in data.columns:
        data['k'] = np.log10(np.abs(data['k']))


    return data
#######################################################################################################################
# Function to load PB data from CSV file
def load_pb_data(file_path):
    dataPB = pd.read_csv(file_path)
    return dataPB
#######################################################################################################################
# Function to make predictions using the trained model
def make_predictions(trained_model, runload_values, n_values, mu_fixed, k_fixed, alphaload_fixed):
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

        # Predict using the trained model
        y_pred = trained_model.predict(input_data)

        # Store the predictions for each value of n
        predictions[n] = y_pred

    return predictions

########################################################################################################################

def plot_filtered_predictions_vs_ground_truth(model, X_train, y_train, y_test, y_test_pred, model_name, export_figure,
                                              y_range=None, save_dir=None, model_specific_name=None):
    """
    Generates a scatter plot comparing predictions to ground truth for both train and test datasets,
    filtered by a specified range of ground truth values.

    Args:
        model: The trained model used for making predictions.
        X_train: Training dataset features.
        y_train: Ground truth values for the training dataset.
        y_test: Ground truth values for the test dataset.
        y_test_pred: Predicted values for the test dataset.
        model_name: Name of the trained model.
        export_figure: Function to save the figure.
        y_range: A tuple specifying the range (min, max) of ground truth values to include in the plot.
    """
    font_size_big = 20
    font_size_small= 20
    # Set default font to Times New Roman globally, including math font
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # Set default font to Times New Roman
        'mathtext.fontset': 'custom',  # Use custom settings for math text
        'mathtext.rm': 'Times New Roman',  # Roman (non-italic) font
        'mathtext.it': 'Times New Roman:italic',  # Italic font
        'mathtext.bf': 'Times New Roman:bold',  # Bold font
        'axes.labelsize': font_size_big,  # Font size for labels
        'xtick.labelsize': font_size_small,  # Font size for x-ticks
        'ytick.labelsize': font_size_small,  # Font size for y-ticks
        'axes.linewidth': 1,  # Width of the axis lines
        'figure.figsize': (7, 6),  # Default figure size
        'legend.fontsize': font_size_small  # Font size for legend
    })

    fig = plt.figure()

    # Predict on training data
    y_train_pred = model.predict(X_train)

    # Apply range filtering if specified
    if y_range is not None:
        y_min_range, y_max_range = y_range

        # Filter for train data
        train_filter = (y_train >= y_min_range) & (y_train <= y_max_range)
        y_train_filtered = y_train[train_filter]
        y_train_pred_filtered = y_train_pred[train_filter]

        # Filter for test data
        test_filter = (y_test >= y_min_range) & (y_test <= y_max_range)
        y_test_filtered = y_test[test_filter]
        y_test_pred_filtered = y_test_pred[test_filter]
    else:
        y_train_filtered = y_train
        y_train_pred_filtered = y_train_pred
        y_test_filtered = y_test
        y_test_pred_filtered = y_test_pred

    # Scatter plot for train data (blue) and test data (red)
    plt.scatter(y_train_filtered, y_train_pred_filtered, alpha=0.5, label='Train Data', color='blue')
    plt.scatter(y_test_filtered, y_test_pred_filtered, alpha=0.5, label='Test Data', color='red')

    # Reference line y=x
    y_min = min(min(y_train_filtered), min(y_test_filtered))
    y_max = max(max(y_train_filtered), max(y_test_filtered))
    plt.plot([y_min, y_max], [y_min, y_max], color='green', linestyle='--', label='y = x')

    # Set labels and title
    plt.xlabel('Ground truth')
    plt.ylabel('Predictions')
    plt.legend()

    # Export the figure
    #export_figure(fig, name=f'RPML_{model_name.lower().replace(" ", "_")}_pre_vs_truth.png', style='presentation_1x1',
    #              savedir= save_dir)
    #export_figure(fig, name=f'{model_specific_name}_{model_name.lower().replace(" ", "_")}_pre_vs_truth.png', style='presentation_1x1',
    #             savedir=save_dir)
    plt.xlim(0, 1)  # Set x-axis limits from 0 to 5
    plt.ylim(0, 1)  # Set x-axis limits from 0 to 5

    plt.tight_layout()
    filename = f'{model_specific_name}_{model_name.lower().replace(" ", "_")}_pre_vs_truth.png'
    if model_name == "XGBoost":
        alphabet = 'b'
    else:
        alphabet = 'a'

    plt.text(0.01, 0.95, f'({alphabet})', fontsize=font_size_big)
    plt.savefig(f"{save_dir}/{filename}")
    # plt.show()
    #####################################################################################################################
def aug_data_provider(data_aug, file_PB='Data_Files/PB_data.csv'):

        # file_PB = 'Data_Files/PB_data.csv'

        # Read the PB data file
        data_PB = pd.read_csv(file_PB)

        # Calculate the 'runload' column based on the given formula
        data_PB['runload'] = ((data_PB['V_pb'] / 2.887) ** (1 / 1.171)) * (3.24 * np.pi ** (2 / 3))

        # Initialize a new column 'PB' in data_aug
        data_aug['PB'] = np.nan

        # Iterate over each row in data_aug
        for index, row in data_aug.iterrows():
            # Get the runload and n value from the current row in data_aug
            runload_BEM = row['runload']
            n_BEM = row['n']

            # Find the closest runload value in data_PB
            closest_idx = find_closest(runload_BEM, np.log10(data_PB['runload']))

            # Construct the column name in data_PB based on the value of n_BEM
            pb_column_name = f'n_{n_BEM}'

            # Check if the column exists in data_PB
            if pb_column_name in data_PB.columns:
                # Retrieve the value from the matching row and column in data_PB
                pb_value = data_PB.loc[closest_idx, pb_column_name]
                # Save the value to the PB column in data_aug
                data_aug.at[index, 'PB'] = pb_value
            else:
                print(f"Column {pb_column_name} not found in data_PB.")

        return data_aug
    ###########################################################################################

def find_closest(runload_value, runload_column):
        idx = (np.abs(runload_column - runload_value)).idxmin()
        return idx

    ###########################################################################################
def append_model_to_file(model_name,num_param, mse_avg, mse_var, r2_avg, r2_var, mse_test, r2_test,
                         save_dir="save_dir_PARPML_6inputs", file_name="latex_table.txt"):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Full file path
    file_path = os.path.join(save_dir, file_name)

    mse_avg = mse_avg *1000
    mse_var = mse_var *1000
    mse_test = mse_test *1000

    # Format the LaTeX-compatible line
    latex_line = (
        f"{model_name} & "
        f"${mse_avg:.4f} \\pm {mse_var:.4f}$ & "
        f"${r2_avg:.4f} \\pm {r2_var:.4f}$ & "
        f"{mse_test:.4f} & "
        f"{r2_test:.4f} & {num_param}\\\\"
    )

    # Append the line to the file
    with open(file_path, "a") as file:  # Open in append mode
        file.write(latex_line + "\n")

    print(f"Added: {latex_line}")
    print(f"Saved to: {file_path}")


def get_trainable_parameters(model_name, model, X_train=None):
    if model_name == 'Linear Regression':
        # For Linear models, number of parameters is features + 1 (for the intercept)
        n_parameters = X_train.shape[1] + 1  # Add 1 for the intercept
        model_name = type(model).__name__

    elif model_name == 'Regression Tree':
        # For Regression Trees, number of parameters is estimated by number of leaf nodes - 1
        n_leaf_nodes = model.get_n_leaves()
        n_parameters = n_leaf_nodes - 1  # Approximation
        model_name = type(model).__name__

    elif model_name == 'Random Forest':
        # For Random Forests, sum of parameters for each tree
        n_trees = model.n_estimators
        n_leaf_nodes_per_tree = model.estimators_[0].get_n_leaves()
        n_parameters = n_trees * (n_leaf_nodes_per_tree - 1)  # Approximation
        model_name = type(model).__name__

    elif model_name == 'XGBoost':
        # For XGBoost, estimated by summing parameters for each tree
        # Get the booster
        booster = model.get_booster()

        # Get the dump of the trees
        trees = booster.get_dump()

        # Function to count parameters in a tree
        def count_params(tree_str):
            splits = len(re.findall(r'f\d+<', tree_str))  # Count number of splits
            leaves = len(re.findall(r'leaf=', tree_str))  # Count number of leaves
            return splits + leaves

        # Count total parameters
        n_parameters = sum(count_params(tree) for tree in trees)

    else:
        return "Model type not supported."

    return n_parameters

##################################################################################################
# Function to compute `a` given delta_JKR
def compute_a(delta_JKR):
    """
    Compute `a` for an array of delta_JKR values using the equation:
    delta_JKR = a^2 - sqrt(2 * a)

    Parameters:
        delta_JKR (array-like): Array or list of delta_JKR values.

    Returns:
        ndarray: Array of computed `a` values.
    """

    # Define the equation to solve
    def equation(a, delta):
        return a ** 2 - np.sqrt(2 * a) - delta

    # Vectorized computation
    a_solutions = []
    for delta in delta_JKR:
        # Solve for each delta_JKR value
        initial_guess = 1.0
        a_solution = fsolve(equation, initial_guess, args=(delta,))
        a_solutions.append(a_solution[0])

    return np.array(a_solutions)


# Function to compute P_JKR given delta_JKR
def compute_P_JKR(delta_JKR):
    """
    Compute `P_JKR` given delta_JKR using the equation:
    P_JKR = 4/3 * a^3 - sqrt(8 * a^3)

    Parameters:
        delta_JKR (array-like): Array or list of delta_JKR values.

    Returns:
        ndarray: Array of computed P_JKR values.
    """
    # Compute `a` for the given delta_JKR values
    a_values = compute_a(delta_JKR)

    # Compute P_JKR using the formula
    P_JKR = (4 / 3) * a_values ** 3 - np.sqrt(8 * a_values ** 3)

    return P_JKR
##################################################################################################
# Function to process the columns in the DataFrame
def process_columns_Seq(data):
    # Check and process 'runload' column if present
    if 'runload' in data.columns and 'mu' in data.columns:
       # data['runload'] = data['runload'] / (data['mu'] * np.pi**(2/3))
        data['runload'] = np.log10(np.abs(data['runload']))

    if 'alphaload' in data.columns and 'mu' in data.columns:
       data['alphaload'] = data['alphaload'] / (data['mu'] * np.pi**(2/3))

    # Check and process 'outputAmplification' column if present
    if 'outputAmplification' in data.columns:
        data['outputAmplification'] = np.log10(np.abs(data['outputAmplification']))
    # Check and process 'k' column if present
    if 'k' in data.columns:
        data['k'] = np.log10(np.abs(data['k']))


    if '0.25' in data.columns:
        data['0.25'] = log_mod(data['0.25'])
    if '0.5' in data.columns:
        data['0.5'] = log_mod(data['0.5'])
    if '0.75' in data.columns:
        data['0.75'] = log_mod(data['0.75'])
    if '0.875' in data.columns:
        data['0.875'] = log_mod(data['0.875'])
    if 'pull-off_force' in data.columns:
        data['pull-off_force'] = log_mod(data['pull-off_force'])

    return data

###############################################################################################

def log_mod(y, epsilon=1e-10):
    """
    Modified logarithm function that handles positive, negative, and zero values.

    Args:
        y (array-like): Input array of values.
        epsilon (float): Small constant to avoid log(0) issues.

    Returns:
        array-like: Transformed array where log is applied to positive and negative values.
    """
    y = np.array(y)  # Ensure input is a NumPy array
    result = np.zeros_like(y)  # Initialize result array

    # Apply log for positive values
    positive_mask = y > 0
    result[positive_mask] = np.log(y[positive_mask] + epsilon)

    # Apply -log for negative values
    negative_mask = y < 0
    result[negative_mask] = -np.log(-y[negative_mask] + epsilon)

    # Zero values remain as 0 (already initialized in `result`)
    return result
##########################################################################################

def inverse_log_mod(result, epsilon=1e-10):
    """
    Inverse of the modified logarithm function `log_mod`.

    Args:
        result (array-like): Transformed array (output of `log_mod`).
        epsilon (float): Small constant used in `log_mod`.

    Returns:
        array-like: Original array before transformation.
    """
    result = np.array(result)  # Ensure input is a NumPy array
    y = np.zeros_like(result)  # Initialize the output array

    # For positive transformed values
    positive_mask = result > 0
    y[positive_mask] = np.exp(result[positive_mask]) - epsilon

    # For negative transformed values
    negative_mask = result < 0
    y[negative_mask] = - (np.exp(-result[negative_mask]) - epsilon)

    # Zero values remain as 0 (already initialized in `y`)
    return y
#########################################################################
