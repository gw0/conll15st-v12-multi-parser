#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Deep learning architecture for PDTB-style discourse parser.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from keras.models import Graph
from keras.layers.core import Activation, Layer, MaskedLayer, TimeDistributedDense, Permute, Dropout, ActivityRegularization
from keras.layers.advanced_activations import ParametricSoftplus
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

from layers.roll import RollOffsets, RepeatVector2, TimeDistributedMerge2, TimeDistributedDense2
from tasks.skipgram import skipgram_model
from tasks.pos import pos_model
from tasks.pdtbmark import pdtbmark_model
from tasks.pdtbpair import pdtbpair_model


def build(max_len, embedding_dim, word2id_size, skipgram_offsets, pos2id_size, pdtbmark2id_size, pdtbpair2id_size, pdtbpair_offsets):

    model = Graph()
    loss = {}

    # input: word ids with masked post-padding (doc, time_pad)
    model.add_input(name='x_word_pad', input_shape=(None,), dtype='int')

    # # input: word ids with random post-padding (doc, time_pad)
    # model.add_input(name='x_word_rand', input_shape=(None,), dtype='int')

    # shared 1: word embedding layer (doc, time_pad, emb)
    model.add_node(Embedding(word2id_size, embedding_dim, input_length=max_len, init='glorot_uniform'), name='shared_1', input='x_word_pad')
    #XXX: mask_zero=True

    # shared 2: forward GRU full sequence layer (doc, time_pad, repr)
    model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_2_gru', input='shared_1')
    #model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_2', input='shared_2_gru')

    # shared 3: forward GRU full sequence layer (doc, time_pad, repr)
    model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_3_gru', input='shared_2_gru')
    # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_3', input='shared_3_gru')

    # shared 4: forward GRU full sequence layer (doc, time_pad, repr)
    model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_4_gru', input='shared_3_gru')
    # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_4', input='shared_4_gru')

    # # shared 5: forward GRU full sequence layer (doc, time_pad, repr)
    # model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_5_gru', input='shared_4_gru')
    # # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_5', input='shared_5_gru')

    # # shared 6: forward GRU full sequence layer (doc, time_pad, repr)
    # model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_6_gru', input='shared_5_gru')
    # # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_6', input='shared_6_gru')

    # # shared 7: forward GRU full sequence layer (doc, time_pad, repr)
    # model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_7_gru', input='shared_6_gru')
    # # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_7', input='shared_7_gru')

    # # shared 8: forward GRU full sequence layer (doc, time_pad, repr)
    # model.add_node(GRU(embedding_dim, return_sequences=True, activation='relu', inner_activation='hard_sigmoid', init='he_uniform', inner_init='orthogonal'), name='shared_8_gru', input='shared_7_gru')
    # # model.add_node(ActivityRegularization(l1=0.01, l2=0.01), name='shared_8', input='shared_8_gru')

    # # skip-gram model: skip-gram labels (doc, time_pad, offset)
    # skipgram_out = skipgram_model(model, ['shared_1', 'x_word_rand'], max_len, embedding_dim, word2id_size, skipgram_offsets)
    # model.add_output(name='y_skipgram', input=skipgram_out)
    # loss['y_skipgram'] = 'mse'

    # # POS model: POS tags (doc, time_pad, pos2id)
    # pos_out = pos_model(model, ['shared_2'], max_len, embedding_dim, pos2id_size)
    # model.add_output(name='y_pos', input=pos_out)
    # loss['y_pos'] = 'binary_crossentropy'

    # PDTB marking model: discourse relation boundary markers (doc, time, offset, pdtbmark2id)
    pdtbmark_out = pdtbmark_model(model, ['shared_4_gru'], max_len, embedding_dim, pdtbmark2id_size)
    model.add_output(name='y_pdtbmark', input=pdtbmark_out)
    loss['y_pdtbmark'] = 'binary_crossentropy'

    # # PDTB pairs model: discourse relation span-pair occurrences (doc, time, offset, pdtbpair2id)
    # pdtbpair_out = pdtbpair_model(model, ['shared_8_gru'], max_len, embedding_dim, pdtbpair2id_size, pdtbpair_offsets)
    # model.add_output(name='y_pdtbpair', input=pdtbpair_out)
    # loss['y_pdtbpair'] = 'binary_crossentropy'

    model.compile(optimizer='rmsprop', loss=loss)
    return model


def get_activations(model, layer, X_batch):
    import theano
    get_activations = theano.function(inputs=[ model.inputs[i].input  for i in model.input_order ], outputs=[ model.nodes[layer].get_output(train=False) ], allow_input_downcast=True)
    activations = get_activations(*[ X_batch[i]  for i in model.input_order ])
    return activations


def test_arch():
    """Basic architecture test."""

    import numpy as np
    import theano
    from keras.utils.visualize_util import plot
    from pprint import pprint

    epochs = 10

    #max_len is computed
    embedding_dim = 2
    word2id_size = 4
    skipgram_window_size = 2
    skipgram_negative_samples = 1
    #skipgram_offsets is computed
    pos2id_size = 3
    pdtbpair2id_size = 16
    pdtb_window_size = 2
    pdtb_negative_samples = 1
    #pdtbpair_offsets is computed

    # all sample data in numeric form
    all_word = [
        [2, 3, 1, 2],
        [1, 2],
    ]
    all_word_len = max([ len(s)  for s in all_word ])
    print("all_word:")
    pprint(all_word)

    all_pos = [
        [2, 1, 1, 2],
        [1, 2],
    ]
    all_pos_len = max([ len(s)  for s in all_pos ])
    print("all_pos:")
    pprint(all_pos)

    # (doc, time, offset, pdtbpair2id_size)
    y_pdtbpair = [
        [
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ],
        [
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ],
    ]

    # compute offsets
    skipgram_offsets = range(-skipgram_window_size // 2, skipgram_window_size // 2 + 1)
    if skipgram_window_size % 2 == 0:
        del skipgram_offsets[skipgram_window_size // 2]
    for i in range(skipgram_negative_samples):
        skipgram_offsets.append(all_word_len + i)
    print("skipgram_offsets: {}".format(skipgram_offsets))

    pdtbpair_offsets = range(-pdtb_window_size // 2, pdtb_window_size // 2 + 1)
    if pdtb_window_size % 2 == 0:
        del pdtbpair_offsets[pdtb_window_size // 2]
    for i in range(pdtb_negative_samples):
        pdtbpair_offsets.append(all_word_len + i)
    print("pdtbpair_offsets: {}".format(pdtbpair_offsets))

    max_len = all_word_len + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)), abs(min(pdtbpair_offsets)), abs(max(pdtbpair_offsets)))
    print("max_len: {}".format(max_len))

    # prepare batch data
    print("prepare batch data")

    # input: word ids with masked and random post-padding (doc, time_pad)
    x_word_pad = []
    x_word_rand = []
    for s in all_word:
        x_word_pad.append(np.hstack([s, np.zeros((max_len - len(s),), dtype=np.int)]))
        x_word_rand.append(np.hstack([s, np.random.randint(1, word2id_size, size=max_len - len(s))]))
    x_word_pad = np.asarray(x_word_pad)
    x_word_rand = np.asarray(x_word_rand)
    print("x_word_pad:")
    pprint(x_word_pad)
    print("x_word_rand:")
    pprint(x_word_rand)

    # output: skip-gram labels (doc, time_pad, offset)
    y_skipgram = []
    for s in x_word_pad:
        #pairs_off = [ zip(s, np.roll(s, -off))  for off in skipgram_offsets ]
        #pairs = map(list, zip(*pairs_off))
        #print pairs

        pairs = [ [ (s[i] != 0 and s[(i + off) % len(s)] != 0)  for off in skipgram_offsets ]  for i in range(max_len) ]
        y_skipgram.append(np.asarray(pairs))
    y_skipgram = np.asarray(y_skipgram)
    print("y_skipgram:")
    pprint(y_skipgram)

    # output: POS tags (doc, time_pad, pos2id_size)
    y_pos = []
    for s in all_pos:
        pos_binary = np.zeros((max_len, pos2id_size), dtype=np.int)
        for i, pos_id in enumerate(s):
            pos_binary[i, pos_id] = 1
        y_pos.append(pos_binary)
    y_pos = np.asarray(y_pos)
    print("y_pos:")
    pprint(y_pos)

    # output: PDTB-style discourse relations (doc, time, offset, pdtbpair2id_size)
    y_pdtbpair = np.asarray(y_pdtbpair, dtype=theano.config.floatX)
    print("y_pdtbpair:")
    pprint(y_pdtbpair)

    # build model
    print("build model")
    model = build(max_len=max_len, embedding_dim=embedding_dim, word2id_size=word2id_size, skipgram_offsets=skipgram_offsets, pos2id_size=pos2id_size, pdtbpair2id_size=pdtbpair2id_size, pdtbpair_offsets=pdtbpair_offsets)
    plot(model, "./model.png")

    # train on batch
    print("train on batch")
    for epoch_i in range(epochs):
        loss = model.train_on_batch({
            'x_word_pad': x_word_pad,
            'x_word_rand': x_word_rand,
            'y_skipgram': y_skipgram,
            'y_pos': y_pos,
            'y_pdtbpair': y_pdtbpair,
        })
        print("[epoch {}] loss: {}".format(epoch_i, loss))


if __name__ == '__main__':
    # attach debugger
    def debugger(type, value, tb):
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        pdb.pm()
    import sys
    sys.excepthook = debugger

    # tests
    test_arch()
