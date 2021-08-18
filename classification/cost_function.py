"""
This module have the cost functions for classification problems involve in NLP
"""
from typing import Dict
import numpy as np

import utils.util


def cost_classification_words(
    word_label: Dict[str, str],
    word_vector: Dict[str, list],
    theta: Dict[str, np.ndarray],
) -> float:
    """
    Compute the cost for cross-entropy classification model.

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
    float
        Cost for the model with the currently theta.

    """
    labels = utils.util.get_labels(word_label)
    cost = 0
    for word, i_label in word_label.items():
        denom = sum(
            [np.exp(np.dot(theta[label], word_vector[word])) for label in labels]
        )
        cost += np.log(np.exp(np.dot(theta[i_label], word_vector[word])) / denom)

    return -(1 / len(word_label)) * cost
