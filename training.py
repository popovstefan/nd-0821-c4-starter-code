import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
if not os.path.exists(model_path):
    os.makedirs(model_path)


# Function for training the model
def train_model():
    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # read the data
    df = pd.read_csv(dataset_csv_path + "/finaldata.csv")
    y = df["exited"]
    X = df.drop(columns=["exited", "corporation"])

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    joblib.dump(model, os.path.join(model_path, "trainedmodel.pkl"))


if __name__ == "__main__":
    train_model()
