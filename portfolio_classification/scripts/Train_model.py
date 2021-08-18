# -*- coding: utf-8 -*-
"""
This scrip is to train and save the classification model
"""
import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import utils.util
import portfolio_classification.product_classifier

# load data
base = "./portfolio_classification/scripts/"
file_name = "portfolio_crawler"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio", "Label_3"]:
    data = data[~data[column].isnull()]
    data = data[data[column] != ""]
data = data.fillna("")

use_cols = [
    "Nombre",
    "Marca",
    "Sub_categoria1",
    "Sub_categoria2",
    "Sub_categoria3",
    "Sub_categoria4",
    "Sub_categoria5",
]

data["phrase"] = utils.util.columns2phrase(data, use_cols)

# Load pre-train word2vec
base = "./glove/scripts/"
file_name = "word2vec"
file_path = os.path.join(base, file_name + ".json")
with open(file_path, "r") as f:
    word2vec = json.load(f)

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

# Get labels
encoder_label = LabelEncoder()
encoder_label.fit(data["Label_3"])
y_labels = encoder_label.transform(data["Label_3"])

# Train
clf = SVC(kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced")
clf.fit(X_features, y_labels)

# Save model
joblib.dump(clf, "./portfolio_classification/classification_model.joblib")
joblib.dump(encoder_label, "./portfolio_classification/encoder_label.joblib")
