"""
This is the documentation for this module
"""
import random
from typing import List
from typing import Optional
from typing import Dict
import re
import string
import numpy as np
import pandas as pd


np.random.seed(0)


def gen_vocabulary(corpus: List[str]) -> List[str]:
    """
    get the corpus vocabulary

    Parameters
    ----------
    corpus : list
        This list contains the tokenized corpus

    Returns
    -------
    list
        unique word list.
    """
    return list(set(corpus))


def gen_theta(
    vocabulary: List[str], dimension: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a vector that will contain the vector representacion
    for each word, both central word and context word, the first half related
    with central words and second part related with context words.

    Parameters
    ----------
    vocabulary : list
        list of unique words in the corpus.
    dimension : int
        Size of dimension that will have each vector representation of words.
    seed : int, optional
        Set random generation. The default is None.

    Returns
    -------
    numpy.ndarray
        Random vector with size: 2 * vocabulary size * dimention, contains the
        vector representation for each word.
    """
    theta_size = 2 * len(vocabulary) * dimension
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(-1, 1, theta_size)


def find_index(word: str, vocabulary: List[str]) -> int:
    """
    Find location of a word in the vocabulary list.

    Parameters
    ----------
    word : str
        word to search.
    vocabulary : list
        list of unique words in the corpus.

    Returns
    -------
    int
        Index value of word in the vocabulary list.
    """
    return vocabulary.index(word)


def find_location(
    word_index: int, theta: np.ndarray, dimension: int, central: bool = True
) -> List[int]:
    """
    Find the location of a word in the theta vector in terms of start index
    and end index.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : numpy.ndarray
        Array that contains the vector representation of words, initially use
        gen_theta to get this parameter.
    dimension : int
        Size of dimension that have each vector representation of words.
    central : bool, optional
        To get central or context word, if the parameter is True, the return
        will be the location in theta for  a central word, in another case the
        result will be for a context word. The default is True.

    Returns
    -------
    list
        List with two elments, first element contain the index where start the
        vector representation of word_index in theta, the second element
        contains the index where end the vector of word_index.
    """
    if central is True:
        start = word_index * dimension
        end = start + dimension
    else:
        start = len(theta) // 2 + word_index * dimension
        end = start + dimension

    return [start, end]


def find_vector(
    word_index: int, theta: np.ndarray, dimension: int, central: bool = True
) -> np.ndarray:
    """
    Extract the vector representation of a word in theta vector.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : numpy.ndarray
        Array that contains the vector representation of words, initially use
        gen_theta to get this parameter.
    dimension : int
        Size of dimension that will have each vector representation of words.
    central : bool, optional
        To get central or context representation, if the parameter is True,
        the return will be the vector representation in theta for
        a central word, in another case the result will be for a context word.
        The default is True.

    Returns
    -------
    numpy.ndarray
        the vector representation in theta for word_index.
    """
    start, end = find_location(word_index, theta, dimension, central)

    return theta[start:end]


def random_dict(
    co_occurrences: Dict[str, int],
    sample_rate: float,
) -> Dict[str, int]:
    """
    Select a random keys (central-context) and filter it in co_occurrences.

    Parameters
    ----------
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.
    sample_rate: float
        To take a sample from cooccurrence.

    Returns
    -------
    Dict[str, int]
        co_occurrency for filter keys, central or context.

    """
    conexions = list(co_occurrences.keys())
    n_samples = int(len(conexions) * sample_rate)
    sample = random.choices(conexions, k=n_samples)

    sample_dict = {choice: co_occurrences[choice] for choice in sample}

    return sample_dict


def get_glove_vectors(
    vocabulary: List[str], theta: np.ndarray, central: bool = False
) -> Dict[str, list]:
    """
    Organize the word vector representation in a dictionary.

    Parameters
    ----------
    vocabulary : list
        list of unique words in the corpus.
    theta : numpy.ndarray
        Array that contains the vector representation of words.
    central : bool, optional
        Especificate filter in terms of central or context words.
        The default is False. it means the default return the context words.

    Returns
    -------
    Dict[str, list]
        Dictionary that the key is the word and the value is the vector representation.

    """

    dimension = len(theta) // 2 // len(vocabulary)
    data = {}
    for word in vocabulary:
        word_index = vocabulary.index(word)
        word_vector = find_vector(word_index, theta, dimension, central=central)
        data[word] = list(word_vector)

    return data


def get_labels(word_label: Dict[str, str]) -> List[str]:
    """
    Obtain the labels related in the model.

    Parameters
    ----------
    word_label : Dict[str, str]
        Dictionary that contain the classification of words, the key represent
        the word and the value is the label or topic.

    Returns
    -------
    List[str]
        This list contain all posible labels in the model given.

    """

    return list(set(word_label.values()))


def gen_theta_class_words(labels: List[str], dimension: int) -> Dict[str, np.ndarray]:
    """
    Create initial parameters theta, represent the weights for model the labels.

    Parameters
    ----------
    labels : List[str]
        This list contain all posible labels in the model given.
    dimension : int
        Size of dimension that will have each vector of weights.

    Returns
    -------
    label_vector = Dict[str, np.ndarray]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.
    """

    return {label: np.random.uniform(-1, 1, dimension) for label in labels}


def gen_grandient(theta: Dict[str, np.ndarray]) -> Dict[str, int]:
    """
    Generate the dictionary in order to save the futures values for the gradient,
    useful to avoid check if the key exist when the gradient is updating.

    Parameters
    ----------
    theta : Dict[str, np.ndarray]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.

    Returns
    -------
    Dict[str, int]
        DESCRIPTION.

    """

    return {label: 0 for label in theta.keys()}


def filter_vocabulary(df: pd.DataFrame, f_vocabulary: List[str]):
    """
    filter words from a Dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe where the keys are the words.
    f_vocabulary : List[str]
        List of words tha want to filter in the dataframe.

    Returns
    -------
    df_ind : TYPE
        DESCRIPTION.

    """
    df_ind = df.loc[df.index.intersection(f_vocabulary)]
    return df_ind


def df_to_dict(dataframe: pd.DataFrame) -> Dict[str, list]:
    """
    transform a data frame to a dictionary where the keys are the index and
    the values are the rows.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    Dict[str, Union[list, str, float]]
        DESCRIPTION.

    """

    dataframe = dataframe.to_dict(orient="index")
    return {word: list(dataframe[word].values()) for word in dataframe.keys()}


def gen_theta_dict(
    vocabulary: List[str], dimension: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create initial parameters theta, this dictionary contain two dictionary,
    first one related with central words and second one refers to context words.

    Parameters
    ----------
    vocabulary : List[str]
        This list contain all vocabulary in the model given.
    dimension : int
        Size of dimension that will have each vector.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.
    """
    np.random.seed(0)

    theta = {
        "central": gen_theta_class_words(vocabulary, dimension),
        "context": gen_theta_class_words(vocabulary, dimension),
    }

    return theta


def gen_grandient_dict(vocabulary: List["str"]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate the dictionary with the same shape of theta..

    Parameters
    ----------
    theta : Dict[str, np.ndarray]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        DESCRIPTION.

    """

    gradient = {
        "central": {label: 0 for label in vocabulary},
        "context": {label: 0 for label in vocabulary},
    }
    return gradient


def del_key(co_occurrences: Dict[str, int], key_del: str) -> Dict[str, int]:
    """
    Detele some key from dictionary.

    Parameters
    ----------
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.
    del_key : str
        Key that want to be deleted.

    Returns
    -------
    None.

    """

    return {
        key: value
        for key, value in co_occurrences.items()
        if key_del not in key.split("<>")
    }


def get_vocabulary(co_occurrences: Dict[str, int]) -> List[str]:
    """

    Create a list with the vocabulary involve in co_occurrence.
    Parameters
    ----------
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.

    Returns
    -------
    List[str]
        list of unique words.

    """
    vocabulary = []
    for key in list(co_occurrences.keys()):
        vocabulary.append(key.split("<>")[0])
        vocabulary.append(key.split("<>")[1])

    return list(set(vocabulary))


def clean_word(word: str) -> str:
    """

    To only take leters and numbers
    Parameters
    ----------
    word : str
        Word that want to clean.

    Returns
    -------
    str
        Word.

    """

    return re.sub("[^A-Za-z0-9]+", "", word).lower()


def clean_phrase(phrase: str) -> str:
    """

    Clean the phrase, punctuation
    ----------
    word : str
        Word that want to clean.

    Returns
    -------
    str
        Word.

    """

    return re.sub("[%s]" % re.escape(string.punctuation), " ", phrase).lower()


def drop_duplicate_key(co_occurrences: Dict[str, int]) -> Dict[str, int]:
    """
    Drop the key when central and context words are the same.

    Parameters
    ----------
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.

    Returns
    -------
    Dict[str, int]
        Return the co-occurrence without duplicate keys.

    """
    return {
        key: value
        for key, value in co_occurrences.items()
        if key.split("<>")[0] != key.split("<>")[1]
    }


def columns2phrase(DataFrame: pd.DataFrame, use_cols: List[str]) -> List[str]:
    """
    Takes a set of columns and concatenate the text for each row.

    Parameters
    ----------
    DataFrame : pd.DataFrame
        DESCRIPTION.
    use_cols : List[str]
        DESCRIPTION.

    Returns
    -------
    List[str]
        DESCRIPTION.

    """
    text_mx = DataFrame[use_cols].values
    return [" ".join(text_mx[i]) for i in range(len(text_mx))]


def metricsreport2df(reports: Dict[str, Dict]) -> pd.DataFrame:
    """
    Transform the metrics report models to dataframe

    Parameters
    ----------
    reports : Dict[str, Dict]
        This dictionary contains the report returns by
        sklearn.metrics.classification_repor, the key correspond to the model.

    Returns
    -------
    DataFrame with the diferent metrics and models.

    """
    dataframe = pd.DataFrame()

    for model in list(reports.keys()):
        df_temp = pd.DataFrame(reports[model]).transpose().reset_index()
        df_temp["model"] = model
        dataframe = pd.concat([dataframe, df_temp])

    dataframe = dataframe.rename(columns={"index": "label/avg"})

    return dataframe
