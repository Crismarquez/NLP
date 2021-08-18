# -*- coding: utf-8 -*-
"""
This module manipulate the words and vectors to get classification features
"""
from random import randrange
from typing import Dict, List
import numpy as np
import pandas as pd

import utils.util


def converter_phrase(
    phrase: str,
    word2vec: Dict[str, np.ndarray],
    dim_word2vec: int,
    phrase_size: int = 21,
    dropout=False,
) -> np.ndarray:
    """
    This function transform a pharse to a embedding tensor, where axis 1
    represent the word according the position in the phrase and axis 2
    represent the vector representation.


    Parameters
    ----------
    phrase : str
        the text to embedding.
    word2vec : Dict[str, np.ndarray]
        word vector representation.
    dim_word2vec : int
        dimension of word vector representation.
    phrase_size : int, optional
        How many words wolud be embedding in the tensor. The default is 21.
    dropout : bool, optional
        delete some word in the phrase, randomly. The default False.

    Returns
    -------
    phrase_vec : np.ndarray
        tensor of rank 3, the dim 0 have only one position, thee dim 1 is
        according to the order of words in the phrase and dim 2 is the word
        vector representation.

    """
    phrase = phrase.split()[:phrase_size]
    if dropout:
        pos_del = randrange(len(phrase))
        phrase[pos_del] = ""

    phrase_vec = np.zeros((1, phrase_size, dim_word2vec))

    for pos, word in enumerate(phrase):
        phrase_vec[0, pos, :] = np.array(word2vec.get(word, [0] * dim_word2vec))

    gap = phrase_size - len(phrase)

    if gap > 0:
        for pos in range(len(phrase), len(phrase) + gap):
            phrase_vec[0, pos, :] = np.array([-10] * dim_word2vec)

    return phrase_vec


def converter_block(
    corpus: List[str],
    word2vec: Dict[str, np.ndarray],
    phrase_size: int = 21,
    dropout=False,
) -> np.ndarray:
    """
    This function transform a set of phrases to a embedding tensor, axis 0
    represent each phrase, the axis 1represent the word according the position
    in the phrase and axis 2 represent the vector representation.

    Parameters
    ----------
    corpus : List[str]
        set of phrases.
    word2vec : Dict[str, np.ndarray]
        word vector representation.
    phrase_size : int, optional
        How many words wolud be embedding in the tensor. The default is 21.

    Returns
    -------
    block_vec : np.ndarray
        tensor of rank 3, the dim 0 have only one position, thee dim 1 is
        according to the order of words in the phrase and dim 2 is the word
        vector representation.

    """
    dim_word2vec = len(word2vec[list(word2vec.keys())[0]])
    block_vec = np.zeros((len(corpus), phrase_size, dim_word2vec))

    for pos, phrase in enumerate(corpus):
        phrase_vec = converter_phrase(
            phrase, word2vec, dim_word2vec, phrase_size=phrase_size, dropout=dropout
        )
        block_vec[pos, :, :] = phrase_vec

    return block_vec


def to_vector(word: str, word2vec: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Transform word to vector representation from word2vec dictionary, if the
    word does not exists the result will be [0].

    Parameters
    ----------
    word : str
        word.
    word2vec : Dict[str, np.ndarray]
        word vector representations.

    Returns
    -------
    np.ndarray
        the vector representition.

    """

    return np.array(word2vec.get(word, [0]))


def weight(position: int, factor: int = 20) -> float:
    """
    Calculate the weight for each position, this weight will be multiplicate
    for word vector.

    Parameters
    ----------
    position : int
        DESCRIPTION.
    factor : int
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return np.exp(-position / factor)


def phrase2vector(phrase: str, word2vec: Dict[str, np.ndarray]) -> np.ndarray:
    """
    This function transform a phrase into a vector representation using
    word2vec dictionary and weight for each word.

    Parameters
    ----------
    phrase : str
        the text to embedding.
    word2vec : Dict[str, np.ndarray]
        word vector representation.

    Returns
    -------
    TYPE
        Vector with dimension dim_word2vec.

    """
    phrase = phrase.split()

    X = np.array(
        [
            to_vector(word, word2vec)
            for word in phrase
            if len(to_vector(word, word2vec)) > 1
        ]
    )
    weights = [weight(i) for i, _ in enumerate(X)]
    X = [weights[i] * v / sum(weights) for i, v in enumerate(X)]

    return np.sum(X, axis=0)


def phrases2vectors(corpus: List[str], word2vec: Dict[str, np.ndarray]) -> np.ndarray:
    """
    This function transform a set of phrases to a embedding vector for each
    phrase, usin the function phrase2vector.

    Parameters
    ----------
    corpus : List[str]
        set of phrases.
    word2vec : Dict[str, np.ndarray]
        word vector representation.

    Returns
    -------
    block_vec : TYPE
        DESCRIPTION.

    """

    block_vec = np.array([phrase2vector(phrase, word2vec) for phrase in corpus])
    return block_vec


def balance_classes(
    dataframe: pd.DataFrame, col_label: str, max_samples: int
) -> pd.DataFrame:
    """
    This function reduce the amount of mayorities classes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DESCRIPTION.
    col_label : str
        DESCRIPTION.
    max_samples : int
        DESCRIPTION.

    Returns
    -------
    DataFrame.

    """
    dataframe["one"] = 1
    volum = dataframe.groupby(col_label)["one"].sum().sort_values().reset_index()
    sampling = volum[volum["one"] > max_samples][col_label].values

    sample = pd.DataFrame()
    for label in sampling:
        sample = pd.concat(
            [sample, dataframe[dataframe[col_label] == label].sample(max_samples)],
            axis=0,
        )

    take_labels = volum[volum["one"] <= max_samples][col_label].values
    sample = pd.concat(
        [sample, dataframe[dataframe[col_label].isin(take_labels)]], axis=0
    )

    return sample


def get_features(
    df_portfolio: pd.DataFrame,
    use_strcols: List[str],
    use_pricecol: str,
    word2vec: Dict[str, List[float]],
) -> np.ndarray:
    """
    This function convert the string and price from portfolio to a features
    for input model classification.

    Parameters
    ----------
    df_portfolio : pd.DataFrame
        DataFrame with scraped portfolio data.
    use_strcols : List[str]
        List with the names of columns to concatenate and generate word2vec.
    use_pricecol: str
        Column with price information.
    word2vec : Dict[str, np.ndarray]
        word vector representation.
    Returns
    -------
    np.ndarray - features.

    """
    df_portfolio["phrase"] = utils.util.columns2phrase(df_portfolio, use_strcols)
    X_features = phrases2vectors(df_portfolio["phrase"].values, word2vec)
    X_price = np.log(df_portfolio[[use_pricecol]].values)
    X_features = np.append(X_features, X_price, axis=1)
    return X_features
