#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
POS tagging model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from keras.layers.core import Activation, TimeDistributedDense
import numpy as np


### Model

def pos_model(model, ins, max_len, embedding_dim, pos2id_size, pre='pos'):
    """POS tagging model as Keras Graph."""

    # POS tag dense neural network (doc, time_pad, pos2id)
    model.add_node(TimeDistributedDense(pos2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build indexes

def build_pos2id(all_words, max_size=None, min_count=0, pos2id=None):
    """Build index for all POS tags (id 0 reserved for masking, id 1 for unknown POS tags)."""
    if pos2id is None:
        pos2id = {}

    # count POS tags occurrences
    pos_cnts = {}
    for doc_id in all_words:
        for word in all_words[doc_id]:
            try:
                pos_cnts[word['PartOfSpeech']] += 1
            except KeyError:
                pos_cnts[word['PartOfSpeech']] = 1

    # ignore POS tags with low occurrences
    for w, cnt in pos_cnts.iteritems():
        if cnt < min_count:
            del pos_cnts[w]

    # rank POS tags by decreasing occurrences and use as index
    pos2id_rev = [None, ""] + sorted(pos_cnts, key=pos_cnts.get, reverse=True)
    if max_size is not None:
        pos2id_rev = pos2id_rev[:max_size]

    # mapping of POS tags to ids
    pos2id.update([ (w, i) for i, w in enumerate(pos2id_rev) ])
    return pos2id


### Encode

def encode_y_pos(doc_ids, all_words, pos2id, word_crop, max_len):
    """Encode POS tags (doc, time_pad, pos2id)."""

    y_pos = []
    for doc_id in doc_ids:
        # map POS tags to ids
        ids = []
        for word in all_words[doc_id][:word_crop]:
            try:
                ids.append(pos2id[word['PartOfSpeech']])
            except KeyError:  # missing in index
                ids.append(pos2id[""])

        # map ids to one-hot encoding
        onehot = np.zeros((max_len, len(pos2id)), dtype=np.int)
        onehot[np.arange(max_len), np.hstack([ids, np.zeros((max_len - len(ids),), dtype=np.int)])] = 1
        y_pos.append(onehot)

    # return as numpy array
    y_pos = np.asarray(y_pos)
    return y_pos
