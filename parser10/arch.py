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

