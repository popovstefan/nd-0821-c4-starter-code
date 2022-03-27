import json
import os
import sys

import deployment
import ingestion
import scoring
import training

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

# Check and read new data

# first, read ingestedfiles.txt
ingested_files = []
with open(os.path.join(prod_deployment_path, "ingestedfiles.csv"), "r") as f:
    for line in f:
        ingested_files.append(line.rstrip())

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
has_not_listed = False
for filename in os.listdir(input_folder_path):
    if input_folder_path + "/" + filename not in ingested_files:
        has_not_listed = True

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not has_not_listed:
    sys.exit(0)

# Checking for model drift
# check whether the score from the deployed model is different
# from the score from the model that uses the newest ingested data
ingestion.merge_multiple_dataframe()
new_f1 = scoring.score_model(prod=True)

with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
    old_f1 = float(f.read())

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 > old_f1:
    sys.exit(0)

training.train_model()
scoring.score_model(prod=True)

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
os.system("python diagnostics.py")
os.system("python reporting.py")
os.system("python apicalls.py")