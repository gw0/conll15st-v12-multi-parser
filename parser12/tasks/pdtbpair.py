#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
PDTB pairs model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

from keras.layers.core import Activation, Permute
import numpy as np

from layers.roll import RollOffsets, RepeatVector2, TimeDistributedDense2
from conll15st_relations import filter_tags, tag_to_rtsns


### Model

def pdtbpair_model(model, ins, max_len, embedding_dim, pdtbpair2id_size, pdtbpair_offsets, pre='pdtbpair'):
    """PDTB pairs model of discourse relation span-pair occurrences as Keras Graph."""

    # repeat word vector (doc, time_pad, repeat, repr)
    model.add_node(RepeatVector2(len(pdtbpair_offsets), axis=2), name=pre + '_repeat', input=ins[0])

    # roll context vector to offsets (doc, time_pad, offset, repr)
    model.add_node(RollOffsets(pdtbpair_offsets, axis=1), name=pre + '_offsets', input=ins[0])

    # dense neural network on word-context pairs (doc, time_pad, offset, pdtbpair2id)
    #1
    #model.add_node(TimeDistributedDense2(pdtbpair2id_size, init='he_uniform'), name=pre + '_dense', inputs=[pre + '_repeat', pre + '_offsets'], merge_mode='concat')
    #model.add_node(Activation('relu'), name=pre + '_act', input=pre + '_dense')

    #2
    model.add_node(TimeDistributedDense2(2 * embedding_dim, init='he_uniform'), name=pre + '_dense2', inputs=[pre + '_repeat', pre + '_offsets'], merge_mode='concat')
    model.add_node(Activation('relu'), name=pre + '_act2', input=pre + '_dense2')
    #model.add_node(Dropout(0.1), name=pre + '_act2', input=pre + '_act2_')
    model.add_node(TimeDistributedDense2(pdtbpair2id_size, init='he_uniform'), name=pre + '_dense', input=pre + '_act2')
    model.add_node(Activation('relu'), name=pre + '_act', input=pre + '_dense')

    #3
    # model.add_node(Permute(dims=(1, 3, 2)), name=pre + '_permute2', inputs=[pre + '_repeat', pre + '_offsets'], merge_mode='concat')
    # model.add_node(TimeDistributedDense2(len(pdtbpair_offsets), init='he_uniform'), name=pre + '_dense2', input=pre + '_permute2')
    # model.add_node(Activation('relu'), name=pre + '_sigmoid2', input=pre + '_dense2')
    # model.add_node(Permute(dims=(1, 3, 2)), name=pre + '_unpermute2', input=pre + '_sigmoid2')
    # model.add_node(TimeDistributedDense2(pdtbpair2id_size, init='he_uniform'), name=pre + '_dense', input=pre + '_unpermute2')
    # model.add_node(Activation('relu'), name=pre + '_act', input=pre + '_dense')
    return pre + '_act'


### Build indexes

def build_pdtbpair2id():
    """Build PDTB discourse relation span-pair index and weights."""

    pdtbpair2id = {
        "Arg1-Arg1": 0,
        "Arg1-Arg2": 1,
        "Arg1-Connective": 2,
        "Arg1-Rest": 3,
        "Arg2-Arg1": 4,
        "Arg2-Arg2": 5,
        "Arg2-Connective": 6,
        "Arg2-Rest": 7,
        "Connective-Arg1": 8,
        "Connective-Arg2": 9,
        "Connective-Connective": 10,
        "Connective-Rest": 11,
        "Rest-Arg1": 12,
        "Rest-Arg2": 13,
        "Rest-Connective": 14,
        "Rest-Rest": 15,
    }
    pdtbpair2id_weights = {
        "Arg1-Arg1": 1.,
        "Arg1-Arg2": 1.,
        "Arg1-Connective": 1.,
        "Arg1-Rest": 0.05,
        "Arg2-Arg1": 1.,
        "Arg2-Arg2": 1.,
        "Arg2-Connective": 1.,
        "Arg2-Rest": 0.05,
        "Connective-Arg1": 1.,
        "Connective-Arg2": 1.,
        "Connective-Connective": 1.,
        "Connective-Rest": 0.05,
        "Rest-Arg1": 0.05,
        "Rest-Arg2": 0.05,
        "Rest-Connective": 0.05,
        "Rest-Rest": 0.,
    }
    return pdtbpair2id, pdtbpair2id_weights


### Encode

def encode_y_pdtbpair(doc_ids, all_words, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, word_crop, max_len, filter_prefixes=None):
    """Encode PDTB-style discourse relations as span-pair occurrences (doc, time, offset, pdtbpair2id)."""

    y_pdtbpair = []
    for doc_id in doc_ids:
        doc_len = len(all_words[doc_id])

        # map word pairs with PDTB-style tags to pairwise occurrences
        pairs = np.zeros((max_len, len(pdtbpair_offsets), len(pdtbpair2id)))
        for w1_i in range(max_len):  # iterate word 1

            # filtered word 1 tags by specified relation tags
            w1_tags = []
            if w1_i < word_crop and w1_i < doc_len:
                w1_tags = all_words[doc_id][w1_i]['Tags'].keys()
                w1_tags = filter_tags(w1_tags, filter_prefixes)

            for off_i, off in enumerate(pdtbpair_offsets):  # iterate word 2
                w2_i = (w1_i + off) % max_len
                pdtbpair2id_key = None

                # filtered word 2 tags by specified relation tags
                w2_tags = []
                if w2_i < word_crop and w2_i < doc_len:
                    w2_tags = all_words[doc_id][w2_i]['Tags'].keys()
                    w2_tags = filter_tags(w2_tags, filter_prefixes)

                # mark occurrences with word 1 and word 2
                for w1_tag in w1_tags:
                    w1_rtype, w1_rsense, w1_rnum, w1_rspan = tag_to_rtsns(w1_tag)

                    # check if word 2 is in same relation
                    pdtbpair2id_key = "{}-Rest".format(w1_rspan)
                    for w2_tag in w2_tags:
                        w2_rtype, w2_rsense, w2_rnum, w2_rspan = tag_to_rtsns(w2_tag)

                        if w1_rtype == w2_rtype and w1_rsense == w2_rsense and w1_rnum == w2_rnum:
                            pdtbpair2id_key = "{}-{}".format(w1_rspan, w2_rspan)
                            break

                    # update pair
                    pairs[w1_i, off_i, pdtbpair2id[pdtbpair2id_key]] += pdtbpair2id_weights[pdtbpair2id_key]

                # else mark occurrences with only word 2
                if not w1_tags:
                    for w2_tag in w2_tags:
                        w2_rtype, w2_rsense, w2_rnum, w2_rspan = tag_to_rtsns(w2_tag)

                        # no word 1 tags
                        pdtbpair2id_key = "Rest-{}".format(w2_rspan)

                        # update pair
                        pairs[w1_i, off_i, pdtbpair2id[pdtbpair2id_key]] += pdtbpair2id_weights[pdtbpair2id_key]

                # else mark no occurrences between word 1 and word 2
                if pdtbpair2id_key is None:
                    pdtbpair2id_key = "Rest-Rest"

                    # update pair
                    pairs[w1_i, off_i, pdtbpair2id[pdtbpair2id_key]] += pdtbpair2id_weights[pdtbpair2id_key]
        y_pdtbpair.append(pairs)

    # return as numpy array
    y_pdtbpair = np.asarray(y_pdtbpair)
    return y_pdtbpair


### Decode

def fitness_partial(pairs, offsets, pair2id, pair2id_weights, sets, update_sets, max_len):
    """Evaluate fitness difference after applying updates to span-pair occurrences."""

    fitness = 0.
    for w1_rspan, w1_set in update_sets.items():  # iterate over updates
        for w1_i in w1_set:
            for off_i, off in enumerate(offsets):  # iterate over offsets
                w2_i = (w1_i + off) % max_len
                for w2_rspan, w2_set in sets.items():  # iterate over sets
                    if w2_i in w2_set:
                        # evaluate w1-w2 pair occurrence
                        pair2id_key = "{}-{}".format(w1_rspan, w2_rspan)
                        fitness += pairs[w1_i, off_i, pair2id[pair2id_key]]

                        # evaluate opposite w2-w1 pair occurrence
                        try:
                            off_j = offsets.index(-off)
                        except ValueError:
                            pass
                        else:
                            pair2id_key = "{}-{}".format(w2_rspan, w1_rspan)
                            fitness += pairs[w2_i, off_j, pair2id[pair2id_key]]
    return fitness


def extract_max_spans(pairs, offsets, pair2id, pair2id_weights, sets, sets_max_len, max_len):
    """Extract max spans from span-pairs occurrences."""

    def explode_set(s, offsets, max_len):
        return set([ (i + off) % max_len  for i in s for off in offsets ])

    todo_set = set([])
    for s in sets.values():
        todo_set.update(explode_set(s, offsets, max_len))
    for s in sets.values():
        todo_set.difference_update(s)

    while todo_set:
        #print "todo_set", todo_set

        # find which word added to which set maximizes fitness
        best_fitness = 0.
        for i in todo_set:
            # try adding to each set
            for k in sets.keys():
                update_sets = {k: set([i])}
                fitness = fitness_partial(pairs, offsets, pair2id, pair2id_weights, sets, update_sets, max_len)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_update_sets = update_sets

        # reached maximal spans
        if best_fitness <= 0.:
            break

        #print "best_fitness", best_fitness, best_update_sets

        # add best word into best set
        for k, s in best_update_sets.items():
            sets[k].update(s)
            if len(sets[k]) >= sets_max_len[k]:  # heuristic for invalid sets
                todo_set = set([])
                break
            todo_set.update(explode_set(s, offsets, max_len))
        for s in sets.values():
            todo_set.difference_update(s)

    # subtract extracted spans
    fitness = 0.
    for w1_rspan, w1_set in sets.items():  # iterate over sets
        for w1_i in w1_set:
            for off_i, off in enumerate(offsets):  # iterate over offsets
                w2_i = (w1_i + off) % max_len
                for w2_rspan, w2_set in sets.items():  # iterate over sets
                    if w2_i in w2_set:
                        # subtract w1-w2 pair occurrence
                        pair2id_key = "{}-{}".format(w1_rspan, w2_rspan)
                        fitness += pairs[w1_i, off_i, pair2id[pair2id_key]]
                        pairs[w1_i, off_i, pair2id[pair2id_key]] -= pair2id_weights[pair2id_key]
    sets['fitness'] = fitness  #XXX
    return sets


def decode_y_pdtbpair(doc_ids, all_words, y_pdtbpair, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, max_len, rtype, rsense):
    """Reconstruct PDTB-style discourse relations from span-pair occurrences."""

    sets_max_len = {  #XXX
        'Arg1': 50,
        'Arg2': 50,
        'Connective': 10,
        'Rest': 100,
    }
    max_relations = 10  #XXX
    def list_compaction(input):  #XXX
        output = []
        first = last = None # first and last number of current consecutive range
        for item in sorted(input):
            if first is None:
                first = last = item # bootstrap
            elif item == last + 1: # consecutive
                last = item # extend the range
            else: # not consecutive
                output.append((first, last)) # pack up the range
                first = last = item
        # the last range ended by iteration end
        output.append((first, last))
        return output

    all_relations = {}
    for d, doc_id in enumerate(doc_ids):
        doc_len = len(all_words[doc_id])

        # iteratively extract relations
        all_relations[doc_id] = []
        while True:

            # find best seeds for Arg1 and Arg2
            best_fitness = 0.
            arg1_seed = -float('inf')
            arg2_seed = -float('inf')

            for i in range(min(doc_len, max_len)):  # iterate word 1
                for off_i, off in enumerate(pdtbpair_offsets):  # iterate word 2
                    j = (i + off) % max_len

                    # find max pair "Arg1-Arg2" or "Arg2-Arg1" as seeds
                    if y_pdtbpair[d][i, off_i, pdtbpair2id["Arg1-Arg2"]] > best_fitness:
                        arg1_seed = i
                        arg2_seed = j
                        best_fitness = y_pdtbpair[d][i, off_i, pdtbpair2id["Arg1-Arg2"]] * pdtbpair2id_weights["Arg1-Arg2"]
                    if y_pdtbpair[d][i, off_i, pdtbpair2id["Arg2-Arg1"]] > best_fitness:
                        arg1_seed = j
                        arg2_seed = i
                        best_fitness = y_pdtbpair[d][i, off_i, pdtbpair2id["Arg2-Arg1"]] * pdtbpair2id_weights["Arg2-Arg1"]

            # no more relations of given relation type:sense
            if best_fitness <= 0.:
                break

            # greedy clustering on pairwise occurrences
            sets = {
                'Arg1': set([arg1_seed]),
                'Arg2': set([arg2_seed]),
                'Connective': set(),
                'Rest': set(),
            }
            sets = extract_max_spans(y_pdtbpair[d], pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, sets, sets_max_len, max_len)

            # build PDTB-style relation
            arg1_token_list = sorted(set([ k  for i in sets['Arg1'] if i < doc_len for k in all_words[doc_id][i]['TokenList'] ]))
            arg2_token_list = sorted(set([ k  for i in sets['Arg2'] if i < doc_len for k in all_words[doc_id][i]['TokenList'] ]))
            conn_token_list = sorted(set([ k  for i in sets['Connective'] if i < doc_len for k in all_words[doc_id][i]['TokenList'] ]))

            relation = {}
            relation['DocID'] = doc_id
            relation['Type'] = rtype
            relation['Sense'] = [rsense]
            relation['Arg1'] = {'TokenList': arg1_token_list}
            relation['Arg2'] = {'TokenList': arg2_token_list}
            relation['Connective'] = {'TokenList': conn_token_list}

            if doc_id in ["wsj_1000", "wsj_2205"]:  #XXX
                print
                print doc_id, len(all_relations[doc_id]), "fitness: {:.4f}".format(sets['fitness']), "sizes:", len(sets['Arg1']), len(sets['Arg2']), len(sets['Connective']), len(sets['Rest'])
                print "arg1_set:", len(arg1_token_list), list_compaction(arg1_token_list)
                print ">", " ".join([ all_words[doc_id][i]['Text']  for i in sorted(sets['Arg1']) if i < doc_len ])
                print "arg2_set:", len(arg2_token_list), list_compaction(arg2_token_list)
                print ">", " ".join([ all_words[doc_id][i]['Text']  for i in sorted(sets['Arg2']) if i < doc_len ])
                print "conn_set:", len(conn_token_list), list_compaction(conn_token_list)
                print ">", " ".join([ all_words[doc_id][i]['Text']  for i in sorted(sets['Connective']) if i < doc_len ])

            all_relations[doc_id].append(relation)
            if len(all_relations[doc_id]) >= max_relations or any([ len(sets[k]) >= sets_max_len[k]  for k in sets_max_len.keys() ]):  # heuristic for invalid relations
                break
    return all_relations
