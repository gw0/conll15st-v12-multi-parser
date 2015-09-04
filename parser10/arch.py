#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Deep learning architecture for PDTB-style discourse parser.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from keras.models import Graph
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def build(vocabulary_n, embedding_dim):
    """Build model with one input and layer-wise outputs."""

    model = Graph()
    model.add_input(name='word', ndim=2)

    # word embedding lookup table
    model.add_node(Embedding(vocabulary_n, embedding_dim), name='embedding', input='word')

    # skip-gram with negative sampling XXX
    skipgram_window = 4
    skipgram_negative = 5
    model.add_node(LSTM(embedding_dim, skipgram_window * embedding_dim), name='skipgram', input='embedding')
    model.add_output(name='skipgram_out', input='skipgram')

    model.compile(optimizer='rmsprop', loss={'skipgram_out':'mse'})
    return model


def init_word2vec(model, word2vec_bin, word2vec_dim):
    """Initialize word embeddings with pre-trained word2vec vectors."""

    pass  #TODO
