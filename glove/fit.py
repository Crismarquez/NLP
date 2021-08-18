# -*- coding: utf-8 -*-
"""
module to update theta parameters
"""
from typing import Dict
import numpy as np


def update_theta(
    theta: Dict[str, Dict[str, np.ndarray]],
    gradient: Dict[str, Dict[str, np.ndarray]],
    learning_rate: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    return update theta, using grandient descent.

    Parameters
    ----------
    theta : Dict[str, Dict[str, np.ndarray]]
        Contains the vector representation of words, the first dictionary
        related with central and the second with context words.
    gradient : Dict[str, Dict[str, np.ndarray]]
        DESCRIPTION.
    learning_rate : float
        DESCRIPTION.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]

    """
    for principal, central_context in gradient.items():
        for word, value in central_context.items():
            theta[principal][word] = theta[principal][word] - learning_rate * value

    return theta
