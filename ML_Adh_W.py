import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np
from matplotlib import pyplot as plt
from cpsme.export_figure import export_figure

# from cpsme.export_figure import export_figure
import joblib
import os
import time

# Here, we import details from utilities
from utilities import latex_table_template
from utilities import process_columns
from utilities import file_path_RPML_5inputs_W
from utilities import save_dir_RPML_5inputs_W
from utilities import plot_filtered_predictions_vs_ground_truth
from utilities import append_model_to_file
from utilities import get_trainable_parameters

# Ensure the directory exists (optional, if you want to create it if it doesn't exist)
save_dir = save_dir_RPML_5inputs_W
os.makedirs(save_dir, exist_ok=True)

# Load the CSV file into a DataFrame
# data = pd.read_csv(file_path_RPML_5inputs)
data = pd.read_csv(file_path_RPML_5inputs_W)

# Call the utility function to process the columns.
# It means we are making some changes in dimensions, please check utilities.py file
data = process_columns(data)

# Load your dataset
input_columns = ['runload', 'alphaload', 'mu', 'k', 'n'] # be carefull  of the figure in the paper
output_column = 'workofSepration'

X = data[input_columns]
y = data[output_column]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Regression Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42,
                                   max_iter=1000)
}

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variables to store results for LaTeX table
latex_rows = []

# Iterate over each model
for model_name, model in models.items():
    mse_folds = []
    r2_folds = []

    # K-Fold Cross-validation
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        y_val_pred = model.predict(X_val_fold)

        # Compute MSE and R² for each fold
        mse_folds.append(mean_squared_error(y_val_fold, y_val_pred))
        r2_folds.append(r2_score(y_val_fold, y_val_pred))

    start_time = time.perf_counter()
    # Final evaluation on test data
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    train_time = end_time - start_time

    y_test_pred = model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Assuming you have the variables `model`, `X_train`, `y_train`, `y_test`, `y_test_pred`, and `model_name`
    # Define the range you want to filter by, for example: y_range =  (0, 1.25) or y_range = None
    y_range = (0, 1.0)
    # y_range = None
    # plot_filtered_predictions_vs_ground_truth(model, X_train, y_train, y_test, y_test_pred, model_name, export_figure,
    #                                          y_range=y_range, save_dir=save_dir_RPML_5inputs)
    plot_filtered_predictions_vs_ground_truth(model, X_train, y_train, y_test, y_test_pred, model_name, export_figure,
                                              y_range=y_range, save_dir=save_dir,
                                              model_specific_name='RPML')

    # Compute averages
    mse_var = np.std(mse_folds)
    mse_avg = np.mean(mse_folds)
    r2_var = np.std(r2_folds)
    r2_avg = np.mean(r2_folds)


    num_param = get_trainable_parameters(model_name, model, X_train)
    append_model_to_file(model_name,num_param, mse_avg, mse_var, r2_avg, r2_var, mse_test, r2_test,save_dir_RPML_5inputs_W, 'RPML_model_comparison_table.txt', train_time=train_time)

# Save each trained model
for model_name, model in models.items():
    save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_model.pkl")
    joblib.dump(model, save_path)
    print(f"{model_name} saved at {save_path}")