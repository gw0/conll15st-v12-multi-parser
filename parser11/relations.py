#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Handle word-pair representation of PDTB-style discourse relations.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse

import conll15st_relations
import conll15st_words


rpart_to_id = {
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


def _triu_to_id(i, j, window_size):
    """Convert 2D upper triangle coordinates to 1D list id."""

    if not (j - window_size <= i and i < j and j >= 1):
        return None
    return i + window_size - j + (j - 1) * window_size


def _id_to_triu(k, window_size):
    """Convert 1D list id to 2D upper triangle coordinates."""
    
    j = k // window_size + 1
    i = (k % window_size) - window_size + j
    return (i, j)


def get_pair_triu(rts_pairs, i, j, window_size):
    """Get pair by 2D upper triangle coordinates."""

    if j < i:  # swap if needed
        i, j = j, i
    if j <= i or i < j - window_size:  # invalid outside window
        return None

    return rts_pairs[_triu_to_id(i, j, window_size)]


def get_pair_id(rts_pairs, k):
    """Get pair by 1D list coordinates."""

    return rts_pairs[k]


def get_pair_value_triu(rts_pairs, i, i_rpart, j, j_rpart, window_size):
    """Get pair value by 2D upper triangle coordinates."""

    if j < i:  # swap if needed
        i, i_rpart, j, j_rpart = j, j_rpart, i, i_rpart
    if j <= i or i < j - window_size:  # outside window
        if i_rpart == "Rest" or j_rpart == "Rest":
            return .01  # non-zero for pair with rest
        else:
            return 0.  # zero for pair without rest

    rpart_key = "{}-{}".format(i_rpart, j_rpart)
    pair = get_pair_triu(rts_pairs, i, j, window_size)
    return pair[rpart_to_id[rpart_key]]


def get_pair_value_id(rts_pairs, k, w1_rpart, w2_rpart):
    """Get pair value by 1D list coordinates."""

    rpart_key = "{}-{}".format(w1_rpart, w2_rpart)
    pair = get_pair_id(rts_pairs, k)
    return pair[rpart_to_id[rpart_key]]


def extract_rts_pairs(rts_tag, doc_words, window_size):
    """Extract word-pairs representation for given relation type:sense."""

    rts_pairs = []
    for _ in range(len(doc_words) * window_size):
        rts_pairs.append([0.] * len(rpart_to_id))

    for j, w2 in enumerate(doc_words[1:], start=1):
        for i in range(j - window_size, j):  # w1 in backward window
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            rpart_key = None

            # match w1 for relation type:sense
            if i >= 0:  # is inside of document
                w1 = doc_words[i]
                for w1_tag in w1['Tags']:
                    w1_rtype, w1_rsense, w1_rnum, w1_rpart = w1_tag.split(":")
                    w1_rts = ":".join([w1_rtype, w1_rsense])
                    w1_rtsn = ":".join([w1_rtype, w1_rsense, w1_rnum])

                    if w1_rts != rts_tag:  # skip other tags
                        continue

                    # is also w2 in same relation
                    rpart_key = "{}-Rest".format(w1_rpart)
                    for w2_tag in w2['Tags']:
                        w2_rtype, w2_rsense, w2_rnum, w2_rpart = w2_tag.split(":")
                        w2_rts = ":".join([w2_rtype, w2_rsense])
                        w2_rtsn = ":".join([w2_rtype, w2_rsense, w2_rnum])

                        if w2_rtsn == w1_rtsn:  # in same relation
                            rpart_key = "{}-{}".format(w1_rpart, w2_rpart)
                            break

                    # update pair
                    pair[rpart_to_id[rpart_key]] += 1.

            # else match only w2 for relation type:sense
            if rpart_key is None:
                for w2_tag in w2['Tags']:
                    w2_rtype, w2_rsense, w2_rnum, w2_rpart = w2_tag.split(":")
                    w2_rts = ":".join([w2_rtype, w2_rsense])
                    w2_rtsn = ":".join([w2_rtype, w2_rsense, w2_rnum])

                    if w2_rts != rts_tag:  # skip other tags
                        continue

                    # no match for w1
                    rpart_key = "Rest-{}".format(w2_rpart)

                    # update pair
                    pair[rpart_to_id[rpart_key]] += 1.

            # if no match for relation type:sense
            if rpart_key is None:
                rpart_key = "Rest-Rest"

                # update pair
                pair[rpart_to_id[rpart_key]] += 1
    return rts_pairs


def to_pairs(words, rts_tags=None, window_size=20):
    """Transform representation from tags to word-pairs by (document id, type:sense)."""

    if rts_tags is None:
        rts_tags = [
            "Explicit:Comparison.Contrast",
            "Explicit:Contingency.Cause.Reason",
            "Explicit:Contingency.Condition",
            "Explicit:Expansion.Conjunction",
        ]

    pairs = {}
    for doc_id in words:
        pairs[doc_id] = {}

        for rts_tag in rts_tags:
            pairs[doc_id][rts_tag] = extract_rts_pairs(rts_tag, words[doc_id], window_size)
    return pairs


def fitness_diff_on(i, rpart, rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size, verbose=False):
    """Estimate fitness difference if word i is assigned a given rpart."""

    fitness = 0.

    # pairs in arg1_set
    for j in arg1_set:
        fitness += get_pair_value_triu(rts_pairs, i, rpart, j, "Arg1", window_size)
    if verbose:
        print fitness

    # pairs in arg2_set
    for j in arg2_set:
        fitness += get_pair_value_triu(rts_pairs, i, rpart, j, "Arg2", window_size)
        if verbose and get_pair_value_triu(rts_pairs, i, rpart, j, "Arg2", window_size) > 0:
            print i, j, fitness
    if verbose:
        print fitness

    # pairs in conn_set
    for j in conn_set:
        fitness += get_pair_value_triu(rts_pairs, i, rpart, j, "Connective", window_size)
    if verbose:
        print fitness

    # pairs in rest_set
    if rpart == "Rest":
        for j in rest_set:
            fitness += get_pair_value_triu(rts_pairs, i, rpart, j, "Rest", window_size)
    if verbose:
        print fitness

    return fitness


def extract_max_relation(rts_tag, rts_pairs, doc_words, window_size):
    """Extract max relation from word-pairs representation for given relation type:sense."""

    # find best Arg1 and Arg2 seeds
    best_fitness = -1
    arg1_seed = -1
    arg2_seed = -1

    for k, pair in enumerate(rts_pairs):
        i, j = _id_to_triu(k, window_size)

        # find max pair "Arg1-Arg2" or "Arg2-Arg1" as seeds
        if pair[rpart_to_id["Arg1-Arg2"]] > best_fitness:
            arg1_seed = i
            arg2_seed = j
            best_fitness = pair[rpart_to_id["Arg1-Arg2"]]
        if pair[rpart_to_id["Arg2-Arg1"]] > best_fitness:
            arg1_seed = j
            arg2_seed = i
            best_fitness = pair[rpart_to_id["Arg2-Arg1"]]

    # no more relations of given relation type:sense
    if arg1_seed < 0 or arg2_seed < 0:
        return None

    # greedy clustering by adding words to sets that maximize fitness
    arg1_set = set([arg1_seed])
    arg2_set = set([arg2_seed])
    conn_set = set()
    rest_set = set()
    missing_set = set(range(len(doc_words))) - arg1_set - arg2_set - conn_set - rest_set

    #print rts_tag
    arg1_token_list = sorted(set([ k  for i in arg1_set for k in doc_words[i]['TokenList'] ]))
    arg2_token_list = sorted(set([ k  for i in arg2_set for k in doc_words[i]['TokenList'] ]))
    conn_token_list = sorted(set([ k  for i in conn_set for k in doc_words[i]['TokenList'] ]))
    #print "arg1_seed:", arg1_set, arg1_token_list, ">", " ".join([ doc_words[i]['Text']  for i in sorted(arg1_set) ])
    #print "arg2_seed:", arg2_set, arg2_token_list, ">", " ".join([ doc_words[i]['Text']  for i in sorted(arg2_set) ])
    #print "conn_seed:", conn_set, conn_token_list, ">", " ".join([ doc_words[i]['Text']  for i in sorted(conn_set) ])
    #print "rest_seed:", rest_set

    while missing_set:
        # find which word added to which set maximizes fitness
        fitness = best_fitness
        best_i = -1
        best_in_set = None

        for i in missing_set:
            if i < min(arg1_set | arg2_set | conn_set) - window_size:
                continue
            if i > max(arg1_set | arg2_set | conn_set) + window_size:
                continue

            # try adding to arg1_set
            fitness_diff = fitness_diff_on(i, "Arg1", rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size)
            if fitness + fitness_diff > best_fitness:
                best_i = i
                best_in_set = arg1_set
                best_fitness = fitness + fitness_diff

            # try adding to arg2_set
            fitness_diff = fitness_diff_on(i, "Arg2", rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size)
            if fitness + fitness_diff > best_fitness:
                best_i = i
                best_in_set = arg2_set
                best_fitness = fitness + fitness_diff

            # try adding to conn_set
            fitness_diff = fitness_diff_on(i, "Connective", rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size)
            if fitness + fitness_diff > best_fitness:
                best_i = i
                best_in_set = conn_set
                best_fitness = fitness + fitness_diff

            # try adding to rest_set
            fitness_diff = fitness_diff_on(i, "Rest", rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size)
            if fitness + fitness_diff > best_fitness:
                best_i = i
                best_in_set = rest_set
                best_fitness = fitness + fitness_diff

        # maximal fitness match around seeds found
        if best_i == -1:
            break

        if best_in_set == arg1_set:
            best_name = "arg1"
        if best_in_set == arg2_set:
            best_name = "arg2"
        if best_in_set == conn_set:
            best_name = "conn"
        if best_in_set == rest_set:
            best_name = "rest"
        #print best_i, best_name, fitness, best_fitness
        #if arg1_seed == 749 and best_in_set != rest_set and doc_words[best_i]['TokenList'][0] < 745:
        #    fitness_diff_on(best_i, "Arg2", rts_pairs, arg1_set, arg2_set, conn_set, rest_set, window_size, verbose=True)
        #    raise Exception("foo")

        # add best word into best set
        missing_set.remove(best_i)
        best_in_set.add(best_i)

    # subtract extracted relation
    for i in arg1_set:
        for j in arg2_set:
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            if not pair:
                continue

            # update pair
            pair[rpart_to_id["Arg1-Arg2"]] = 0.
            pair[rpart_to_id["Arg2-Arg1"]] = 0.

    for i in arg1_set:
        for j in rest_set:
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            if not pair:
                continue

            # update pair
            pair[rpart_to_id["Arg1-Rest"]] = 0.
            pair[rpart_to_id["Rest-Arg1"]] = 0.

    for i in arg2_set:
        for j in rest_set:
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            if not pair:
                continue

            # update pair
            pair[rpart_to_id["Arg2-Rest"]] = 0.
            pair[rpart_to_id["Rest-Arg2"]] = 0.

    for i in arg1_set:
        for j in arg1_set:
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            if not pair:
                continue

            # update pair
            pair[rpart_to_id["Arg1-Arg1"]] -= 1.

    for i in arg2_set:
        for j in arg2_set:
            pair = get_pair_triu(rts_pairs, i, j, window_size)
            if not pair:
                continue

            # update pair
            pair[rpart_to_id["Arg2-Arg2"]] -= 1.

    # new relation
    doc_id = doc_words[0]['DocID']
    rtype, rsense = rts_tag.split(":")
    arg1_token_list = sorted(set([ k  for i in arg1_set for k in doc_words[i]['TokenList'] ]))
    arg2_token_list = sorted(set([ k  for i in arg2_set for k in doc_words[i]['TokenList'] ]))
    conn_token_list = sorted(set([ k  for i in conn_set for k in doc_words[i]['TokenList'] ]))

    relation = {}
    relation['DocID'] = doc_id
    relation['Type'] = rtype
    relation['Sense'] = [rsense]
    relation['Arg1'] = {'TokenList': arg1_token_list}
    relation['Arg2'] = {'TokenList': arg2_token_list}
    relation['Connective'] = {'TokenList': conn_token_list}

    #print "arg1_set:", arg1_set, arg1_token_list, "\n>", " ".join([ doc_words[i]['Text']  for i in sorted(arg1_set) ])
    #print "arg2_set:", arg2_set, arg2_token_list, "\n>", " ".join([ doc_words[i]['Text']  for i in sorted(arg2_set) ])
    #print "conn_set:", conn_set, conn_token_list, "\n>", " ".join([ doc_words[i]['Text']  for i in sorted(conn_set) ])
    #print "rest_set:", rest_set
    #print

    return relation


def to_relations(pairs, words, window_size=20):
    """Transform representation from word-pairs to relations by document id."""

    relations = {}
    for doc_id in pairs:
        relations[doc_id] = []

        for rts_tag in pairs[doc_id]:
            # extract all relations
            while True:
                relation = extract_max_relation(rts_tag, pairs[doc_id][rts_tag], words[doc_id], window_size)
                if relation is None:
                    break
                else:
                    relations[doc_id].append(relation)
    return relations


if __name__ == '__main__':
    # attach debugger
    def debugger(type, value, tb):
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        pdb.pm()
    import sys
    sys.excepthook = debugger

    # parse arguments
    argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    argp.add_argument('dataset_dir',
        help="CoNLL15st dataset directory")
    argp.add_argument('doc_ids', nargs='*',
        help="Filter by document ids")
    args = argp.parse_args()

    # load all relations by document id
    relations_all = conll15st_relations.load_relations_all(args.dataset_dir)
    relations_all = conll15st_relations.conv_tokenlists(relations_all)
    relations_all = conll15st_relations.conv_sensenum(relations_all)

    # load all words by document id
    words_all = conll15st_words.load_words_all(args.dataset_dir)
    words_all = conll15st_relations.conv_linkers_to_tags(words_all, relations_all)

    # list all document ids
    doc_ids = [ doc_id  for doc_id in words_all if doc_id in relations_all ]

    if args.doc_ids:
        doc_ids = args.doc_ids
        print("filter: {}".format(doc_ids))
        relations_all = dict([ (i, relations_all[i]) for i in doc_ids ])
        words_all = dict([ (i, words_all[i]) for i in doc_ids ])

    # transform to word-pairs by document id
    print("transform to word-pairs...")
    rts_tags = {}
    for doc_id in relations_all:
        for relation in relations_all[doc_id]:
            #if relation['Type'] not in ['Implicit']:
            #    continue

            rts = ":".join([relation['Type'], relation['Sense'][0]])  # only first sense
            rts_tags[rts] = 1
    print rts_tags
    pairs = to_pairs(words_all, rts_tags)

    # transform back to relations by (document id, type, sense)
    print("transform back to relations...")
    #relations2_all = to_relations(pairs, words_all)

    # verify results (exact match)
    print("verify results...")
    for doc_id in relations_all:

        print
        #print(doc_id)
        #print("transform back to relations...")
        relations2_all = to_relations({ doc_id: pairs[doc_id] }, { doc_id: words_all[doc_id] })
        #print("verify results...")

        cnt_match = 0
        cnt_not_doc_id = 0
        cnt_not_type = 0
        cnt_not_sense = 0
        cnt_not_arg1 = 0
        cnt_not_arg2 = 0
        cnt_not_connective = 0

        for r1 in relations_all[doc_id]:
            for r2 in relations2_all[doc_id]:
                if r1['DocID'] != r2['DocID']:
                    cnt_not_doc_id += 1
                    continue
                if r1['Type'] != r2['Type']:
                    cnt_not_type += 1
                    continue
                if r1['Sense'][0] != r2['Sense'][0]:  # only first sense
                    cnt_not_sense += 1
                    continue
                if r1['Arg1']['TokenList'] != r2['Arg1']['TokenList']:
                    cnt_not_arg1 += 1
                    continue
                if r1['Arg2']['TokenList'] != r2['Arg2']['TokenList']:
                    cnt_not_arg2 += 1
                    continue
                if r1['Connective']['TokenList'] != r2['Connective']['TokenList']:
                    cnt_not_connective += 1
                    continue
                cnt_match += 1
                break

        print(doc_id)
        #print("match: {} (extr: {}) (orig: {})".format(cnt_match, sum([ len(r)  for r in relations2_all.itervalues() ]), sum([ len(r)  for r in relations_all.itervalues() ])))
        print("match: {} (extr: {}) (orig: {})".format(cnt_match, len(relations2_all[doc_id]), len(relations_all[doc_id])))
        #print("not_doc_id: {} (should be 0)".format(cnt_not_doc_id))
        #print("not_type: {}".format(cnt_not_type))
        #print("not_sense: {}".format(cnt_not_sense))
        #print("not_arg1: {}".format(cnt_not_arg1))
        #print("not_arg2: {} (should be 0)".format(cnt_not_arg2))
        #print("not_connective: {} (should be 0)".format(cnt_not_connective))
