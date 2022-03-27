import json
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from diagnostics import model_predictions

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config["output_model_path"])


# Function for reporting
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    y_true, y_pred = model_predictions(None)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
    disp.ax_.figure.savefig(os.path.join(model_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()
