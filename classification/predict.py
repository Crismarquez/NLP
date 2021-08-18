# -*- coding: utf-8 -*-
"""
Funtions to predict using optimized theta
"""
from typing import Dict, List
import numpy as np


def predict_classwords(
    word_label: Dict[str, str],
    word_vector: Dict[str, List[float]],
    theta: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Predict the categories for each word in word_label, using the optimazed theta.

    Parameters
    ----------
    word_label : Dict[str, str]
        Dictionary that contain the classification of words, the key represent
        the word and the value is the label or topic.
    word_vector : Dict[str, list]
        Contain the word vector representation from glove.
        key for word, value for the vector.
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.

    Returns
    -------
    Dict[str, Dict[str, float]]
        return all values for each category, the key is the word and the value
        is a dictionary with all categories.

    """

    prediction = {}
    for word in word_label.keys():
        prediction[word] = {
            key: np.dot(word_vector[word], theta_val)
            for key, theta_val in theta.items()
        }

    return prediction


def summary_predict_classwords(
    word_label: Dict[str, str],
    word_vector: Dict[str, List[float]],
    theta: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """


    Parameters
    ----------
    word_label : Dict[str, str]
        Contain the classification of words, the key represent the word and
        the value is the label or topic.
    word_vector : Dict[str, list]
        Contain the word vector representation from glove.
        key for word, value for the vector.
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.
    Returns
    -------
    Dict[str, str]
        Contain the word and the prediction of label.

    """

    prediction = predict_classwords(word_label, word_vector, theta)
    summary_prediction = {}
    for word, value in prediction.items():
        max_key = max(value, key=value.get)
        summary_prediction[word] = max_key

    return summary_prediction


def accuracy_classwords(
    word_label: Dict[str, str], summary_prediction: Dict[str, str]
) -> float:
    """
    measure the succesfull cases in prediction dictionary vs total cases.

    Parameters
    ----------
    word_label : Dict[str, str]
        Contain the classification of words, the key represent the word and
        the value is the label or topic.
    summary_prediction : Dict[str, str]
        Contain the word and the prediction of label..

    Returns
    -------
    float
        positive / total.

    """
    return sum(
        1 for key, value in summary_prediction.items() if value == word_label[key]
    ) / len(summary_prediction)
