# -*- coding: utf-8 -*-
"""
This script shows how to use classification model
"""
import os
import json
import pandas as pd
import joblib

import portfolio_classification.product_classifier

# load data
base = "./portfolio_classification/scripts/"
file_name = "portfolio_crawler"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio"]:
    data = data[~data[column].isnull()]
    data = data[data[column] != ""]
data = data.fillna("")


# Load pre-train word2vec
base = "./glove/scripts/"
file_name = "word2vec"
file_path = os.path.join(base, file_name + ".json")
with open(file_path, "r") as f:
    word2vec = json.load(f)

use_strcols = [
    "Nombre",
    "Marca",
    "Sub_categoria1",
    "Sub_categoria2",
    "Sub_categoria3",
    "Sub_categoria4",
    "Sub_categoria5",
]
use_pricecol = "Precio"

X_features = portfolio_classification.product_classifier.get_features(
    data, use_strcols, use_pricecol, word2vec
)

encoder = joblib.load("./portfolio_classification/encoder_label.joblib")
model = joblib.load("./portfolio_classification/classification_model.joblib")

categories = encoder.inverse_transform(model.predict(X_features))

data["prediction"] = categories
data.to_excel("portafolio_clasificado.xlsx", index=False)
