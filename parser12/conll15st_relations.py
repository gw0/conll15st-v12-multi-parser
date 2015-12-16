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


### Tag converters

def rts_to_tag(relation):
    """Convert relation type and sense to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # only first sense
    return ":".join([rtype, rsense])


def rtsns_to_tag(relation, rspan):
    """Convert relation type, sense, sense number, and span to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # only first sense
    rnum = relation['SenseNum'][0]  # only first sense
    return ":".join([rtype, rsense, rnum, rspan])


def tag_to_rtsns(tag):
    """Convert tag to relation type, sense, sense number, and span."""

    rtype, rsense, rnum, rspan = tag.split(":")
    return rtype, rsense, rnum, rspan


def filter_tags(tags, prefixes=None):
    """Filter list of relation tags matching specified prefixes."""

    if prefixes is not None:
        # filter by specified relation tag prefixes
        tags = [ t  for t in tags if any([ t.startswith(p)  for p in prefixes ]) ]
    return tags


### CoNLL15st relations

def load_relations(dataset_dir, relations_ffmt=None):
    """Load all PDTB-style discourse relations by document id.

    Example output:

        all_relations[doc_id][0] = {
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
    all_relations = {}
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
            all_relations[relation['DocID']].append(relation)
        except KeyError:
            all_relations[relation['DocID']] = [relation]
    f.close()
    return all_relations


def conv_span_tokenlist_format(all_relations):
    """Convert all token lists from detailed/gold to token id format and add min/max token id and count.

    Example output:

        all_relations[doc_id][0] = {
            'Arg1': {'CharacterSpanList': [[2493, 2517]], 'RawText': 'and told ...', 'TokenList': [465, 466, ...]},
            'Arg2': {'CharacterSpanList': [[2526, 2552]], 'RawText': "they're ...", 'TokenList': [472, 473, ...]},
            'Connective': {'CharacterSpanList': [[2518, 2525]], 'RawText': 'because', 'TokenList': [471]},
            'TokenMin': 465,
            'TokenMax': 476,
            'TokenCount': 12,
            ...
        }
    """

    for doc_id in all_relations:
        for relation in all_relations[doc_id]:
            # convert from detailed/gold to token id format
            relation['Arg1']['TokenList'] = [ t[2]  for t in relation['Arg1']['TokenList'] ]
            relation['Arg2']['TokenList'] = [ t[2]  for t in relation['Arg2']['TokenList'] ]
            relation['Connective']['TokenList'] = [ t[2]  for t in relation['Connective']['TokenList'] ]

            # add token id min and max and token count
            token_list = sum([ relation[span]['TokenList']  for span in ['Arg1', 'Arg2', 'Connective'] ], [])
            relation['TokenMin'] = min(token_list)
            relation['TokenMax'] = max(token_list)
            relation['TokenCount'] = len(token_list)
    return all_relations


def add_relation_sensenum(all_relations, filter_prefixes=None):
    """Add enumerated relations sense numbers ordered in decreasing token count.

    Example output:

        all_relations[doc_id][0] = {
            ...
            'SenseNum': [1],
        }
    """

    # order and filter relations
    relations_ord = {}
    for doc_id in all_relations:
        # order by increasing token count
        all_relations[doc_id].sort(key=lambda r: r['TokenCount'])

        relations_ord[doc_id] = []
        rnums = {}
        for relation in all_relations[doc_id]:
            # add sense count number
            rnum_key = rts_to_tag(relation)
            try:
                rnums[rnum_key] += 1
            except KeyError:
                rnums[rnum_key] = 1
            relation['SenseNum'] = [str(rnums[rnum_key])]

            tags = [ rtsns_to_tag(relation, span)  for span in ['Arg1', 'Arg2', 'Connective'] ]
            if filter_tags(tags, filter_prefixes):  # relation matches
                relations_ord[doc_id].append(relation)
    return relations_ord


def add_relation_tags(all_words, all_relations):
    """Convert linkers format from CoNLL15st words to relation tags.

    Example output:

        words[doc_id][0] = {
            ...,
            'Linkers': ["arg1_14890"],
            'Tags': {"Explicit:Expansion.Conjunction:4:Arg1": 1},
        }
    """

    linker_to_span = {"arg1": 'Arg1', "arg2": 'Arg2', "conn": 'Connective'}

    # convert linker ids to relation tags on each word
    for doc_id in all_words:
        for word in all_words[doc_id]:
            word['Tags'] = {}
            for linker in word['Linkers']:  # get linker ids for each word
                linker_span, relation_id = linker.split("_")
                span = linker_to_span[linker_span]

                # find by relation id
                for relation in all_relations[doc_id]:
                    if relation_id == str(relation['ID']):  # found relation id
                        tag = rtsns_to_tag(relation, span)
                        try:
                            word['Tags'][tag] += 1  # weird, but in CoNLL15st train dataset
                        except KeyError:
                            word['Tags'][tag] = 1
                        break  # only once
    return all_words


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    argp.add_argument('dataset_dir',
        help="CoNLL15st dataset directory")
    args = argp.parse_args()

    # iterate through all CoNLL15st relations by document id
    all_relations = load_relations(args.dataset_dir)
    all_relations = conv_span_tokenlist_format(all_relations)
    all_relations = add_relation_sensenum(all_relations)
    for doc_id in all_relations:
        for relation in all_relations[doc_id]:
            print(relation)
