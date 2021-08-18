# -*- coding: utf-8 -*-
"""
module to update theta parameters
"""
from typing import Dict
import numpy as np


def update_theta(
    theta: Dict[str, np.ndarray], gradient: Dict[str, np.ndarray], learning_rate: float
) -> Dict[str, np.ndarray]:
    """


    Parameters
    ----------
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.
    gradient : Dict[str, np.ndarray]
        DESCRIPTION.
    learning_rate : float
        DESCRIPTION.

    Returns
    -------
    Dict[str, np.ndarray].

    """

    return {
        key: (value - learning_rate * gradient[key]) for key, value in theta.items()
    }
