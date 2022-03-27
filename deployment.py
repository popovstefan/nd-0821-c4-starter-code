import json
import os

# Load config.json and correct path variable
import shutil

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

# Function for deployment
def store_model_into_pickle():
    """
    Copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    shutil.copy2(src=os.path.join(dataset_csv_path, "ingestedfiles.csv"),
                 dst=os.path.join(prod_deployment_path, "ingestedfiles.csv"))
    shutil.copy2(src=os.path.join(model_path, "trainedmodel.pkl"),
                 dst=os.path.join(prod_deployment_path, "trainedmodel.pkl"))
    shutil.copy2(src=os.path.join(model_path, "latestscore.txt"),
                 dst=os.path.join(prod_deployment_path, "latestscore.txt"))


if __name__ == "__main__":
    store_model_into_pickle()
