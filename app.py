import json
import os

from flask import Flask, request

# Set up variables for use in our script
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """call the prediction function you created in Step 3"""
    # add return value for prediction outputs
    dataset_path = request.json.get('dataset_path')
    y_pred, _ = model_predictions(dataset_path)
    return str(y_pred)


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """check the score of the deployed model"""
    # add return value (a single F1 score number)
    return str(score_model(False))


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """check means, medians, and modes for each column"""
    # return a list of all calculated summary statistics
    return str(dataframe_summary())


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """check timing and percent NA values"""
    # add return value for all diagnostics
    return str("execution_time:" + execution_time() + "\nmissing_data;" + missing_data() + "\noutdated_packages:" + outdated_packages_list())


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
