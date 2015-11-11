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


def build():

    vocab_size = 10000
    embedding_dim = 3

    skipgram_window = 2
    relations_window = 2

    model = Graph()

    # inputs
    model.add_input(name='x_word', input_shape=(None,), dtype='int')
    model.add_input(name='x_rand', input_shape=(None,))

    # word embedding lookup table (shared layer 1)
    model.add_node(Embedding(vocab_size, embedding_dim), name='layer_1', input='x_word')

    # skip-gram model context embedding lookup table
    model.add_node(Embedding(vocab_size, embedding_dim), name='context_emb', inputs=['x_context', 'x_rand'], merge_mode='concat')

    # skip-gram model word-context pairs in window
    skipgram_window

    # skip-gram model dot product
    dot product
    model.add_node(Merge([], mode='mul'), name='skipgram', input='skipgram_pair')

    # forward LSTM (shared layer 2)
    model.add_node(LSTM(embedding_dim, embedding_dim), name='layer_2', input='layer_1')

    # POS tag dense neural networks
    model.add_node(Dense(2 * embedding_dim), name='pos', input='layer_2')

    # discourse relations word-word pairs in window
    relation_window

    # discourse relations dense neural network
    model.add_node(Dense(embedding_dim), name='relation', input='relation_pair')

    # outputs
    model.add_output(name='y_skipgram', input='skipgram')
    model.add_output(name='y_pos', input='pos')
    model.add_output(name='y_relation', input='relation')

    model.compile(optimizer='rmsprop', loss={'y_skipgram':'mse'})
    return model


def buildx(vocabulary_n, embedding_dim):
    """Build model with one input and layer-wise outputs."""

    model = models.Sequential()
    model.add(recurrent.SimpleRNN(x_dim, inner_dim, init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=True))
    #model.add(core.Dropout(0.5))
    #model.add(recurrent.SimpleRNN(inner_dim, inner_dim, init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=True))
    #model.add(core.Dropout(0.5))
    #model.add(recurrent.SimpleRNN(inner_dim, y_dim, init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=True))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.get_config(verbose=1)
    from keras.utils.dot_utils import Grapher
    grapher = Grapher()
    grapher.plot(model, "{}/model.png".format(args.model_dir))

    # alternative snippets
    #model.add(embeddings.Embedding(max_features, x_dim))
    #model.add(recurrent.SimpleRNN(x_dim, inner_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None, truncate_gradient=-1, return_sequences=False))
    #recurrent.SimpleDeepRNN recurrent.GRU recurrent.LSTM
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #loss = lambda y_true, y_pred: T.max(abs(y_true - y_pred)) + T.mean((y_true - y_pred) ** 2)
    #model.compile(loss=loss, optimizer='adam')


def init_word2vec(model, word2vec_bin, word2vec_dim):
    """Initialize word embeddings with pre-trained word2vec vectors."""

    pass  #TODO


    # prepare mapping vocabulary to word2vec vectors
    map_word2vec = joblib.load("./ex02_model/map_word2vec.dump")  #XXX

    # prepare numeric form
    x = []
    y = []
    doc_ids = []
    for doc_id, doc in words.iteritems():
        doc_x = []
        doc_y = []
        for word in doc:
            # map text to word2vec
            try:
                doc_x.append(map_word2vec[word['Text']])
            except KeyError:  # missing in vocab
                doc_x.append(np.zeros(word2vec_dim))

            # map tags to vector
            tags = [0.] * len(tag_to_j)
            for tag, count in word['Tags'].iteritems():
                tags[tag_to_j[tag]] = float(count)
            doc_y.append(tags)

            #print word['Text'], word['Tags']
            #print word['Text'], doc_x[-1][0:1], doc_y[-1]

        x.append(np.asarray(doc_x, dtype=theano.config.floatX))
        y.append(np.asarray(doc_y, dtype=theano.config.floatX))
        doc_ids.append(doc_id)
        if doc_id not in relations:
            relations[doc_id] = []

