# Purpose: Create confusion matrix using predictions on the test set. Note: Modify the number of class labels depending on the test set.
# Execution: python3 confusion_matrix.py -p pickle_file -m trained_model

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

import pandas as pd
import pickle


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pickle_file", required=True, help="input pickle file")
ap.add_argument("-m", "--model_name", required=True, help="what model was used to train?")

args = vars(ap.parse_args())
pickle_file = str(args['pickle_file'])
model_name = str(args['model_name'])

predictions = pickle.load(open(pickle_file, 'rb'))

actual = predictions['gold']
predicted = predictions['pred']

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4])

cm_display.plot()
plt.show()
fname = pickle_file.split('/')[-1]
plt.savefig(f"{model_name}-{fname.replace('.pkl', '.png')}")
