# -*- coding: utf-8 -*-
"""
This script is to create clasification models and evaluate the perfomance for
each one
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

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
file_name = "word2vec"
file_path = os.path.join(base, file_name + ".json")
with open(file_path, "r") as f:
    word2vec = json.load(f)

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)

encoder_category = LabelEncoder()
encoder_category.fit(data["Label"])
labels = list(encoder_category.classes_)
encoder_label = encoder_category.transform(list(encoder_category.classes_))
y = encoder_category.transform(data["Label"])

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# models

# SVM
model_svm = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm.fit(X_train, y_train)

y_predict = model_svm.predict(X_test)

metrics.classification_report(
    y_test, y_predict, labels=encoder_label, target_names=labels, output_dict=True
)

print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label, target_names=labels
    )
)

# Level 2
data = data[~data["Label_2"].isnull()]
data = data[data["Label_2"] != ""]


# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data["phrase"].values, word2vec
)
print("Shape of features vectors:", X_features.shape)

encoder_category_lv2 = LabelEncoder()
encoder_category_lv2.fit(data["Label_2"])
labels_lv2 = list(encoder_category_lv2.classes_)
encoder_label_lv2 = encoder_category_lv2.transform(list(encoder_category_lv2.classes_))

print("Categories: ", labels_lv2)
y_lv2 = encoder_category_lv2.transform(data["Label_2"])

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_lv2, test_size=0.2, random_state=42
)


# models

# SVM
model_svm = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm.fit(X_train, y_train)

y_predict = model_svm.predict(X_test)
print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv2, target_names=labels_lv2
    )
)


# Level 3
data = data[~data["Label_3"].isnull()]
data = data[data["Label_3"] != ""]

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
model_svm = SVC(
    kernel="linear", C=1, decision_function_shape="ovo", class_weight="balanced"
)
model_svm.fit(X_train, y_train)
y_predict = model_svm.predict(X_test)

print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)

# eval televisores
data_2 = data[data["Label_2"] == "Audio y video-televisores"]

# Get features based in word2vec and weigths
X_features = portfolio_classification.product_classifier.phrases2vectors(
    data_2["phrase"].values, word2vec
)

X_price = np.log(data_2[["Precio"]].values)

X_features = np.append(X_features, X_price, axis=1)
print("Shape of features vectors:", X_features.shape)

y_predict = model_svm.predict(X_features)

predict_label = encoder_category_lv3.inverse_transform(y_predict)

pd.DataFrame.from_dict(
    {
        "descrip": data_2["phrase"].values,
        "price": data_2["Precio"].values,
        "label": data_2["Label_3"].values,
        "label_p": predict_label,
    }
).to_excel("prediccion_televisores.xlsx")

len(data_2["Label_3"].values)
len(predict_label)
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
model_svm = SVC(
    kernel="linear", C=1, class_weight="balanced", decision_function_shape="ovo"
)
model_svm.fit(X_train, y_train)

y_predict = model_svm.predict(X_test)

print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv3, target_names=labels_lv3
    )
)


us = NearMiss(ratio=0.5, n_neighbors=3, version=2, random_state=1)
X_train_res, y_train_res = us.fit_sample(X_train, y_train)


# oversampling
os = RandomOverSampler(ratio=0.5)
X_train_os, y_train_os = os.fit_sample(X_train, y_train)

# stragy sub-sampling
us = NearMiss(version=2)

X_train_us, y_train_us = us.fit_resample(X_train, y_train)

# SVM
model_svm = SVC(
    kernel="linear", C=1, class_weight="balanced", decision_function_shape="ovo"
)
model_svm.fit(X_train, y_train)

y_predict = model_svm.predict(X_test)

reports = {}
reports["SVC_linear"] = metrics.classification_report(
    y_test,
    y_predict,
    labels=encoder_label_lv2,
    target_names=labels_lv2,
    output_dict=True,
)
print(
    metrics.classification_report(
        y_test, y_predict, labels=encoder_label_lv2, target_names=labels_lv2
    )
)

# unstructure facebook text
text = [
    "celular iphone 11 pro",
    "laptop mackbook 8gb oferta",
    "tv led 5k ultra hd",
    "plancha para el cabello",
    "audifonos earpods apple iphone lighting 7 8 x plus original",
    "iphone",
    "saludo, ¿quién tiene a la venta un portatil con procesador core i3 o core i5 (de 4th generación hacia arriba)?, económico. Para compra inmediata.",
    "se vende computador.... monitor dell, torre samsung con quemador, viene con el mueble... En perfecto estado",
    "parlantes pequeños económicos Totalmente nuevos, con su respectiva factura, hacemos domicilios",
    "samsung a51 4 de ram 128 de almacenamiento sin cargador en buen estado detalles de uso",
    "iphone x de 64 gb libre para cualquier operador y libre de imei",
    "compro realme xt o 6 o 7",
    "cambio a iphone de 7 en adelante.. el celular esta en perfecto estado todo le funciona 6 meses de comprado como nuevo",
    "busco televisor barato smart tv con netflix",
]
for phrase in text:
    text2vec = portfolio_classification.product_classifier.phrase2vector(
        phrase, word2vec
    ).reshape(1, -1)
    print(
        phrase,
        ": ",
        encoder_category_lv2.inverse_transform(model_svm.predict(text2vec)),
    )
