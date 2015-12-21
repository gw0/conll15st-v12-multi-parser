#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
PDTB marking model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from keras.layers.core import Activation, TimeDistributedDense
import numpy as np

from conll15st_relations import filter_tags, tag_to_rtsns


### Model

def pdtbmark_model(model, ins, max_len, embedding_dim, pdtbmark2id_size, pre='pdtbmark'):
    """PDTB marking model as Keras Graph."""

    # PDTB marking dense neural network (doc, time_pad, pdtbmark2id)
    model.add_node(TimeDistributedDense(pdtbmark2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build indexes

def build_pdtbmark2id(mode='IO'):
    """Build PDTB discourse relation boundary markers index."""

    if mode == 'IOBES':
        pdtbmark2id = {
            "O": [0],
            "B-Arg1": [1],
            "I-Arg1": [2],
            "E-Arg1": [3],
            "S-Arg1": [4],
            "B-Arg2": [5],
            "I-Arg2": [6],
            "E-Arg2": [7],
            "S-Arg2": [8],
            "B-Connective": [9],
            "I-Connective": [10],
            "E-Connective": [11],
            "S-Connective": [12],
        }
    elif mode == 'BE':
        pdtbmark2id = {
            "O": [0],
            "B-Arg1": [1],
            "I-Arg1": [0],
            "E-Arg1": [2],
            "S-Arg1": [1, 2],
            "B-Arg2": [3],
            "I-Arg2": [0],
            "E-Arg2": [4],
            "S-Arg2": [3, 4],
            "B-Connective": [5],
            "I-Connective": [0],
            "E-Connective": [6],
            "S-Connective": [5, 6],
        }
    elif mode == 'IO':
        pdtbmark2id = {
            "O": [0],
            "B-Arg1": [1],
            "I-Arg1": [1],
            "E-Arg1": [1],
            "S-Arg1": [1],
            "B-Arg2": [2],
            "I-Arg2": [2],
            "E-Arg2": [2],
            "S-Arg2": [2],
            "B-Connective": [3],
            "I-Connective": [3],
            "E-Connective": [3],
            "S-Connective": [3],
        }
    pdtbmark2id_size = max([ i  for l in pdtbmark2id.values() for i in l ]) + 1
    return pdtbmark2id, pdtbmark2id_size


### Encode

def encode_y_pdtbmark(doc_ids, all_words, pdtbmark2id, pdtbmark2id_size, word_crop, max_len, filter_prefixes=None):
    """Encode PDTB boundary markers (doc, time_pad, pdtbmark2id)."""

    y_pdtbmark = []
    for doc_id in doc_ids:
        doc_len = len(all_words[doc_id])

        # mark PDTB argument and connective boundaries
        marks = np.zeros((max_len, pdtbmark2id_size), dtype=np.int)
        last_tags = set([])
        for w1_i in range(max_len):  # iterate word 1

            # filtered word 1 tags by specified relation tags
            w1_tags = []
            if w1_i < word_crop and w1_i < doc_len:
                w1_tags = all_words[doc_id][w1_i]['Tags'].keys()
                w1_tags = filter_tags(w1_tags, filter_prefixes)

            # filtered word 2 (next word) tags by specified relation tags
            w2_tags = []
            w2_i = w1_i + 1
            if w2_i < word_crop and w2_i < doc_len:
                w2_tags = all_words[doc_id][w2_i]['Tags'].keys()
                w2_tags = filter_tags(w2_tags, filter_prefixes)

            # mark
            for tag in w1_tags:
                rtype, rsense, rnum, rspan = tag_to_rtsns(tag)

                if tag not in last_tags and tag not in w2_tags:
                    # mark single tag
                    marks[w1_i, pdtbmark2id["S-{}".format(rspan)]] = 1
                elif tag not in last_tags:
                    # mark begin tag
                    marks[w1_i, pdtbmark2id["B-{}".format(rspan)]] = 1
                elif tag not in w2_tags:
                    # mark end tag
                    marks[w1_i, pdtbmark2id["E-{}".format(rspan)]] = 1
                else:
                    # mark inside tag
                    marks[w1_i, pdtbmark2id["I-{}".format(rspan)]] = 1

            # mark other tag
            if not np.any(marks[w1_i]):
                marks[w1_i, pdtbmark2id["O"]] = 1

            last_tags = set(w1_tags)
        y_pdtbmark.append(marks)

    # return as numpy array
    y_pdtbmark = np.asarray(y_pdtbmark)
    return y_pdtbmark
