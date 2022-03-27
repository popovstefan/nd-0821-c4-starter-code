import json
import os

import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000"

# Call each API endpoint and store the responses
headers = {
    'Content-type': 'application/json',
    'Accept': 'text/plain'
}
responses = [
    requests.post("%s/prediction" % URL, json={"dataset_path": "testdata.csv"}, headers=headers).text,
    requests.get("%s/scoring" % URL, headers=headers).text,
    requests.get("%s/summarystats" % URL, headers=headers).text,
    requests.get("%s/diagnostics" % URL, headers=headers).text,
]

# combine all API responses
responses = "\n".join(responses)

# write the responses to your workspace
with open('config.json', 'r') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as f:
    f.write(responses)
