# -*- coding: utf-8 -*-
"""
This script shows the benefits to use word embeding
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import utils.util
import portfolio_classification.product_classifier

# load data
base = "./portfolio_classification/scripts/"
file_name = "portfolio_crawler"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio", "Label", "Label_2", "Label_3"]:
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

# Level 3
# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
print("Shape of features vectors:", X_features.shape)

# add feature price
X_price = np.log(data[["Precio"]].values)

X_features = np.append(X_features, X_price, axis=1)
print("Shape of features vectors:", X_features.shape)

encoder_category_lv3 = LabelEncoder()
encoder_category_lv3.fit(data["Label_3"])
labels_lv3 = list(encoder_category_lv3.classes_)
encoder_label_lv3 = encoder_category_lv3.transform(list(encoder_category_lv3.classes_))

print("Categories: ", labels_lv3)
y_lv3 = encoder_category_lv3.transform(data["Label_3"])
print("n labels = ", len(labels_lv3))

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_lv3, test_size=0.2, random_state=42
)

# models
# SVM
model_svm_0 = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm_0.fit(X_train, y_train)
y_predict = model_svm_0.predict(X_test)

print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# load random word2vec - word2vec pre-train improve resutls
vocabulary = list(word2vec.keys())
theta = utils.util.gen_theta_dict(vocabulary, dimension=100)

word2vec_r = {}
for word in vocabulary:
    word2vec_r[word] = list(theta["context"][word] + theta["central"][word])

# Get features based in random word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec_r
)
print("Shape of features vectors:", X_features.shape)

# add feature price
X_price = np.log(data[["Precio"]].values)

X_features = np.append(X_features, X_price, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_lv3, test_size=0.2, random_state=42
)

# SVM
model_svm_r0 = SVC(
    kernel="linear", C=1, class_weight="balanced", decision_function_shape="ovo"
)
model_svm_r0.fit(X_train, y_train)

y_predict = model_svm_r0.predict(X_test)

print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# train whitout the word 'laptop' using word2vec

# load data
base = "./portfolio_classification/scripts/"
file_name = "portfolio_withoutlaptop"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio", "Label", "Label_2", "Label_3"]:
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
y_labels = encoder_category_lv3.transform(data["Label_3"])

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

# SVM
model_svm_1 = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm_1.fit(X_features, y_labels)


# Get features based in random word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec_r
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

# SVM
model_svm_r1 = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm_r1.fit(X_features, y_labels)

# load data for test
base = "./portfolio_classification/scripts/"
file_name = "portfolio_onlylaptop"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio", "Label", "Label_2", "Label_3"]:
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

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

y_true_1 = encoder_category_lv3.transform(data["Label_3"])
y_predict_1 = model_svm_1.predict(X_features)

print(
    metrics.classification_report(
        y_true_1, y_predict_1, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# Get features based in random word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec_r
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

y_true_r1 = encoder_category_lv3.transform(data["Label_3"])
y_predict_r1 = model_svm_r1.predict(X_features)

print(
    metrics.classification_report(
        y_true_r1, y_predict_r1, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# new portfolio
base = "./portfolio_classification/scripts/"
file_name = "new_portfolio"
file_path = os.path.join(base, file_name + ".xlsx")
data = pd.read_excel(file_path)

# filter valid data
for column in ["Nombre", "Precio", "Label_3"]:
    data = data[~data[column].isnull()]
    data = data[data[column] != ""]

use_cols = [
    "Nombre",
    "Sub_categoria1",
    "Sub_categoria2",
    "Sub_categoria3",
    "Sub_categoria4",
    "Sub_categoria5",
]

for col in use_cols:
    data[col] = data[col].astype(str)

data["phrase"] = utils.util.columns2phrase(data, use_cols)

X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)
y_true = encoder_category_lv3.transform(data["Label_3"])
y_predict = model_svm_0.predict(X_features)

print(
    metrics.classification_report(
        y_true, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec_r
)
X_price = np.log(data[["Precio"]].values)
X_features = np.append(X_features, X_price, axis=1)

y_predict = model_svm_r0.predict(X_features)

print(
    metrics.classification_report(
        y_true, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)
