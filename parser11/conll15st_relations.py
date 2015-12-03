#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
CoNLL15st loading of PDTB-style discourse relations from 'pdtb-data.json'.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import json


def rts_to_tag(relation):
    """Convert relation type and sense to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # only first sense
    return ":".join([rtype, rsense])


def rtsnp_to_tag(relation, rpart):
    """Convert relation type, sense, sense number, and part to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # only first sense
    rnum = relation['SenseNum'][0]  # only first sense
    return ":".join([rtype, rsense, rnum, rpart])


def load_relations_all(dataset_dir, relations_ffmt=None):
    """Load all PDTB-style discourse relations by document id.

    Example output:

        relations_all[doc_id][0] = {
            'Arg1': {'CharacterSpanList': [[2493, 2517]], 'RawText': 'and told ...', 'TokenList': [[2493, 2496, 465, 15, 8], [2497, 2501, 466, 15, 9], ...]},
            'Arg2': {'CharacterSpanList': [[2526, 2552]], 'RawText': "they're ...", 'TokenList': [[2526, 2530, 472, 15, 15], [2530, 2533, 473, 15, 16], ...]},
            'Connective': {'CharacterSpanList': [[2518, 2525]], 'RawText': 'because', 'TokenList': [[2518, 2525, 471, 15, 14]]},
            'DocID': 'wsj_1000',
            'ID': 15007,
            'Type': 'Explicit',
            'Sense': ['Contingency.Cause.Reason'],
        }
    """
    if relations_ffmt is None:
        relations_ffmt = "{}/pdtb-data.json"

    # load all relations
    relations_json = relations_ffmt.format(dataset_dir)
    f = open(relations_json, 'r')
    relations_all = {}
    for line in f:
        relation = json.loads(line)

        # fix inconsistent structure
        if 'TokenList' not in relation['Arg1']:
            relation['Arg1']['TokenList'] = []
        if 'TokenList' not in relation['Arg2']:
            relation['Arg2']['TokenList'] = []
        if 'TokenList' not in relation['Connective']:
            relation['Connective']['TokenList'] = []

        # store by document id
        try:
            relations_all[relation['DocID']].append(relation)
        except KeyError:
            relations_all[relation['DocID']] = [relation]
    f.close()
    return relations_all


def conv_tokenlists(relations_all):
    """Convert all token lists from detailed/gold to token id format.

    Example output:

        relations_all[doc_id][0] = {
            'Arg1': {'CharacterSpanList': [[2493, 2517]], 'RawText': 'and told ...', 'TokenList': [465, 466, ...]},
            'Arg2': {'CharacterSpanList': [[2526, 2552]], 'RawText': "they're ...", 'TokenList': [472, 473, ...]},
            'Connective': {'CharacterSpanList': [[2518, 2525]], 'RawText': 'because', 'TokenList': [471]},
            'TokenMin': 465,
            'TokenMax': 476,
            'TokenCount': 12,
            ...
        }
    """

    for doc_id in relations_all:
        for relation in relations_all[doc_id]:
            # convert from detailed/gold to token id format
            relation['Arg1']['TokenList'] = [ t[2]  for t in relation['Arg1']['TokenList'] ]
            relation['Arg2']['TokenList'] = [ t[2]  for t in relation['Arg2']['TokenList'] ]
            relation['Connective']['TokenList'] = [ t[2]  for t in relation['Connective']['TokenList'] ]

            # add token id min and max and token count
            token_list = sum([ relation[part]['TokenList']  for part in ['Arg1', 'Arg2', 'Connective'] ], [])
            relation['TokenMin'] = min(token_list)
            relation['TokenMax'] = max(token_list)
            relation['TokenCount'] = len(token_list)
    return relations_all


def conv_sensenum(relations_all, filter_tags=None):
    """Enumerate all relations sense numbers and order by decreasing token count.

    Example output:

        relations_all[doc_id][0] = {
            ...
            'SenseNum': [1],
        }
    """

    # order and filter relations
    relations_ord = {}
    for doc_id in relations_all:
        # order by increasing token count
        relations_all[doc_id].sort(key=lambda r: r['TokenCount'])

        relations_ord[doc_id] = []
        rnums = {}
        for relation in relations_all[doc_id]:
            # add sense count number
            rnum_key = rts_to_tag(relation)
            try:
                rnums[rnum_key] += 1
            except KeyError:
                rnums[rnum_key] = 1
            relation['SenseNum'] = [str(rnums[rnum_key])]

            if filter_tags is None:
                # no filter
                relations_ord[doc_id].append(relation)
            else:
                # filter by specified relation tags
                for rpart in ['Arg1', 'Arg2', 'Connective']:
                    if rtsnp_to_tag(relation, rpart) in filter_tags:  # relation found
                        relations_ord[doc_id].append(relation)
                        break  # only once
    return relations_ord


def conv_linkers_to_tags(words_all, relations_all):
    """Convert all linkers on CoNLL15st corpus words to relation tags.

    Example output:

        words[doc_id][0] = {
            ...,
            'Linkers': ["arg1_14890"],
            'Tags': {"Explicit:Expansion.Conjunction:4:Arg1": 1},
        }
    """

    lpart_to_rpart = {"arg1": "Arg1", "arg2": "Arg2", "conn": "Connective"}

    # convert linkers to relation tags on each word
    for doc_id in words_all:
        for word in words_all[doc_id]:
            word['Tags'] = {}
            for linker in word['Linkers']:  # get relation ids for each word
                lpart, rid = linker.split("_")
                rpart = lpart_to_rpart[lpart]

                # find by relation id
                for relation in relations_all[doc_id]:
                    if rid == str(relation['ID']):  # found relation id
                        tag = rtsnp_to_tag(relation, rpart)
                        try:
                            word['Tags'][tag] += 1
                        except KeyError:
                            word['Tags'][tag] = 1
                        break  # only once
    return words_all


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    argp.add_argument('dataset_dir',
        help="CoNLL15st dataset directory")
    args = argp.parse_args()

    # iterate through all CoNLL15st relations by document id
    relations_all = load_relations_all(args.dataset_dir)
    relations_all = conv_tokenlists(relations_all)
    relations_all = conv_sensenum(relations_all)
    for doc_id in relations_all:
        for relation in relations_all[doc_id]:
            print(relation)
