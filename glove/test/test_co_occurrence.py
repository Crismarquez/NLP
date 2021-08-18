# -*- coding: utf-8 -*-
"""
test co_occurence
"""
import glove.co_occurrence


def test_cooccurrences_exists():
    """
    Test a cooccurrence that exist in the vocabulary given.

    Returns
    -------
    None.

    """
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]

    S_WINDOW = 3
    cooccurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)
    assert cooccurrences["i<>like"] == 5


def test_cooccurrences_max_conexion():
    """
    Test a cooccurrence that not exist in the vocabulary given.

    Returns
    -------
    None.

    """
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]

    vocabulary = set(corpus)
    S_WINDOW = 3
    cooccurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)
    assert len(cooccurrences) <= len(vocabulary) * len(vocabulary)


def test_cooccurrences_exists_prev():
    """
    Test a cooccurrence that exist in the vocabulary given.

    Returns
    -------
    None.

    """
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]

    prev_cooccurrence = {"i<>like": 5}
    S_WINDOW = 3
    cooccurrences = glove.co_occurrence.cooccurrences(
        corpus, S_WINDOW, prev_cooccurrence
    )
    assert cooccurrences["i<>like"] == 10
