# -*- coding: utf-8 -*-
"""
This module compute de co-occurrence between two words in a corpus given.
"""
from typing import List, Dict, Optional
import numpy as np
import utils.util


def cooccurrences(
    corpus: List[str],
    s_window: int,
    prev_cooccurrences: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Calculates the frequency for each combination between central and context
    word, return a dictionary where the key is a string that contain the central
    word in the right and context word on the left, this words are separed by
    "<>" character.
    Ignoring the words that not exist in vocabulary.

    Parameters
    ----------
    corpus : List[str]
        This list contains the tokenized corpus.
    s_window : int
        Size of window to take the context words, e.g. s_window = 2, take two
        context words at left and two context words at right.
    prev_cooccurrences: Optional[Dict[str, int]]
        To load previous co_occurrence dictionary. the key is a string that
        contain the central word and context word separed by"<>" character.
    Returns
    -------
    co_occurrences : Dict[str, int]
        Dictionary, the key is a string that contain the central word and
        context word separed by"<>" character.

    """

    co_occurrences = {}
    if prev_cooccurrences is not None:
        co_occurrences = prev_cooccurrences

    for pos_central in range(s_window, len(corpus) - s_window):

        central_word = corpus[pos_central]
        for i in range(s_window):
            context_word_l = corpus[pos_central - (1 + i)]  # left
            context_word_r = corpus[pos_central + (1 + i)]  # right

            co_occurrences[(central_word + "<>" + context_word_l)] = (
                co_occurrences.get(central_word + "<>" + context_word_l, 0) + 1
            )

            co_occurrences[(central_word + "<>" + context_word_r)] = (
                co_occurrences.get(central_word + "<>" + context_word_r, 0) + 1
            )

    return co_occurrences


def matrix_frequency(
    corpus: List[str],
    vocabulary: List[str],
    s_window: int,
) -> np.ndarray:
    """
    Calculates the frequency for each combination between central and context
    word, return a matrix where the columns refers to central words and the rows
    refers to context word. Ignoring the words that not exist in vocabulary.

    Parameters
    ----------
    corpus : List[str]
        This list contains the tokenized corpus.
    vocabulary : List[str]
        List of unique words in the corpus. use gen_vocabulary function.
    s_window : int
        Size of window to take the context words, e.g. s_window = 2, take two
        context words at left and two context words at right.
    Returns
    -------
    matrix numpy.ndarray
        Matrix with row and column size equal to the number of words, each
        element represent the frequency in the corpus for the conexion central
        word (column) and context word(row).
    """
    mtx_frequency = np.zeros((len(vocabulary), len(vocabulary)), "int")

    for pos_central in range(s_window, len(corpus) - s_window):

        if corpus[pos_central] in vocabulary:

            central_index = utils.util.find_index(corpus[pos_central], vocabulary)

            # find conexion in couple (left - right)
            for i in range(s_window):
                take_l = pos_central - (1 + i)  # left
                take_r = pos_central + (1 + i)  # right

                if corpus[take_l] in vocabulary:
                    context_index = utils.util.find_index(corpus[take_l], vocabulary)
                    mtx_frequency[context_index, central_index] = (
                        mtx_frequency[context_index, central_index] + 1
                    )

                if corpus[take_r] in vocabulary:
                    context_index = utils.util.find_index(corpus[take_r], vocabulary)
                    mtx_frequency[context_index, central_index] = (
                        mtx_frequency[context_index, central_index] + 1
                    )

    return mtx_frequency
