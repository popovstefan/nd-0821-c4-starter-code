import json
import os
# Load config.json and get environment variables
import subprocess
import sys
import time

import joblib
import pandas as pd

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config["output_model_path"])


# Function to get model predictions
def model_predictions():
    """Read the deployed model and a test dataset, calculate predictions"""
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    model = joblib.load(os.path.join(model_path, "trainedmodel.pkl"))
    return model.predict(df.drop(columns=["exited", "corporation"]))


# Function to get summary statistics
def dataframe_summary():
    """calculate summary statistics here"""
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    num_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    result = []
    for num_col in num_cols:
        result.append([num_col, "mean", df[num_col].mean()])
        result.append([num_col, "mean", df[num_col].mean()])
        result.append([num_col, "mean", df[num_col].mean()])
    # return value should be a list containing all summary statistics
    return result


def missing_data():
    """Checks for missing data"""
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        result.append([column, str(int(count_na / (count_not_na + count_na) * 100)) + "%"])
    return str(result)


# Function to get timings
def execution_time():
    """calculate timing of training.py and ingestion.py"""
    result = []
    for script in ["training.py", "ingestion.py"]:
        start = time.time()
        os.system(command="python %s" % script)
        end = time.time()
        result.append([script, end - start])
    # return a list of 2 timing values in seconds
    return result


# Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
