import glob
import json
import os

import pandas as pd

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """Function for data ingestion."""
    # laod all *.csv files
    data_sets = glob.glob(input_folder_path + "/*.csv")
    # initialize log and intermediate dataframe list
    dfs = []
    log = []
    # loop over each data set and append results
    for data_set in data_sets:
        log.append(data_set)
        dfs.append(pd.read_csv(data_set))
    # concatenate all dataframes
    df = pd.concat(dfs)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # create output directory
    os.makedirs(output_folder_path)
    # save final dataframe result
    df.to_csv(output_folder_path + "/finaldata.csv", index=False)
    # save log
    with open(output_folder_path + "/ingestedfiles.csv", "w") as f:
        for item in log:
            f.write("%s\n" % item)


if __name__ == '__main__':
    merge_multiple_dataframe()
