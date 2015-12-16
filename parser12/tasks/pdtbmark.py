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

def pdtbmark_model(model, ins, pdtbmark2id_size, pre='pdtbmark'):
    """PDTB marking model as Keras Graph."""

    # PDTB marking dense neural network (doc, time_pad, pdtbmark2id)
    model.add_node(TimeDistributedDense(pdtbmark2id_size), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build indexes

def build_pdtbmark2id():
    """Build PDTB discourse relation boundary markers index."""

    pdtbmark2id = {
        "B-Arg1": 0,
        "E-Arg1": 1,
        "B-Arg2": 2,
        "E-Arg2": 3,
        "B-Connective": 4,
        "E-Connective": 5,
        "": 6,
    }
    return pdtbmark2id


### Encode

def encode_y_pdtbmark(doc_ids, all_words, pdtbmark2id, word_crop, max_len, filter_prefixes=None):
    """Encode PDTB boundary markers (doc, time_pad, pdtbmark2id)."""

    y_pdtbmark = []
    for doc_id in doc_ids:
        doc_len = len(all_words[doc_id])

        # mark PDTB argument and connective boundaries
        marks = np.zeros((max_len, len(pdtbmark2id)), dtype=np.int)
        last_tags = set([])
        for w1_i in range(max_len):  # iterate word 1

            # filtered word 1 tags by specified relation tags
            w1_tags = []
            if w1_i < word_crop and w1_i < doc_len:
                w1_tags = all_words[doc_id][w1_i]['Tags'].keys()
                w1_tags = filter_tags(w1_tags, filter_prefixes)

            # mark begin boundaries
            for tag in set(w1_tags).difference(last_tags):
                rtype, rsense, rnum, rspan = tag_to_rtsns(tag)
                marks[w1_i, pdtbmark2id["B-{}".format(rspan)]] = 1

            # mark end boundaries
            for tag in last_tags.difference(set(w1_tags)):
                rtype, rsense, rnum, rspan = tag_to_rtsns(tag)
                marks[w1_i - 1, pdtbmark2id["E-{}".format(rspan)]] = 1
                marks[w1_i, pdtbmark2id[""]] = 0  # correction

            # mark other
            if not np.any(marks[w1_i]):
                marks[w1_i, pdtbmark2id[""]] = 1

            last_tags = set(w1_tags)
        y_pdtbmark.append(marks)

    # return as numpy array
    y_pdtbmark = np.asarray(y_pdtbmark)
    return y_pdtbmark
