import json
import os

import joblib
import pandas as pd
# Load config.json and get path variables
from sklearn import metrics

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config["output_model_path"])


# Function for model scoring
def score_model():
    """
    This function should take a trained model, load test data,
     and calculate an F1 score for the model relative to the test data.
    It should write the result to the latestscore.txt file
    """
    # load data
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y = df["exited"]
    X = df.drop(columns=["exited", "corporation"])
    # load model
    model = joblib.load(os.path.join(model_path, "trainedmodel.pkl"))
    # predict on test data
    y_pred = model.predict(X)
    # score prediction
    f1 = metrics.f1_score(y, y_pred)
    # save the result
    with open(os.path.join(model_path, "latestscore.txt"), "w") as f:
        f.write("%s\n" % f1)


if __name__ == "__main__":
    score_model()
