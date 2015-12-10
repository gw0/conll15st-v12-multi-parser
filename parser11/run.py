#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
PDTB-style discourse parser (CoNLL15st format).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import csv
import logging
import os
import resource
import time
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
from keras.utils.visualize_util import plot

import arch
import conll15st_relations
import conll15st_words
from conll15st_scorer import scorer


### Logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

class Profiler(object):
    """Helper for monitoring time and memory usage."""

    def __init__(self, log):
        self.log = log
        self.time_0 = None
        self.time_1 = None
        self.mem_0 = None
        self.mem_1 = None
        self.start()

    def start(self):
        self.time_0 = time.time()
        self.mem_0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def stop(self):
        self.time_1 = time.time()
        self.mem_1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.print_usage()

    def print_usage(self):
        self.log.error("(time {:.3f}s, memory {:+.1f}MB, total {:.3f}GB)".format(self.time_1 - self.time_0, (self.mem_1 - self.mem_0) / 1024.0, self.mem_1 / 1024.0 / 1024.0))

def profile(func, log=None):
    """Decorator for monitoring time and memory usage."""

    if log is None:
        log = logging.getLogger(func.__module__)
    profiler = Profiler(log)

    def wrap(*args, **kwargs):
        profiler.start()
        res = func(*args, **kwargs)
        profiler.stop()
        return res

    return wrap


### Performance stats and history

class Stats(object):
    """Handler for performance stats and history."""

    def __init__(self, experiment, train_dir, valid_dir):
        self.fieldnames = ['experiment', 'train_dir', 'valid_dir', 'epoch', 'loss_avg', 'loss_min', 'loss_max', 'train_precision', 'train_recall', 'train_f1', 'valid_precision', 'valid_recall', 'valid_f1', 'time_1', 'time_2', 'time_3']

        self.experiment = experiment
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.history = []

    def append(self, row):
        row['experiment'] = self.experiment
        row['train_dir'] = self.train_dir
        row['valid_dir'] = self.valid_dir
        self.history.append(row)

    def load(self, fname):
        f = open(fname, 'rb')
        freader = csv.DictReader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.history = []
        for row in freader:
            self.append(row)
        f.close()

    def save(self, fname):
        f = open(fname, 'wb')
        fwriter = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fwriter.writeheader()
        for row in self.history:
            fwriter.writerow(row)
        f.close()


### Load CoNLL15st dataset

def load_conll15st(dataset_dir, filter_prefixes=None):
    """Load CoNLL15st dataset in original form."""

    # load all relations by document id
    all_relations = conll15st_relations.load_relations(dataset_dir)
    all_relations = conll15st_relations.conv_span_tokenlist_format(all_relations)
    all_relations = conll15st_relations.add_relation_sensenum(all_relations, filter_prefixes=filter_prefixes)

    # load all words by document id
    all_words = conll15st_words.load_words(dataset_dir)
    all_words = conll15st_relations.add_relation_tags(all_words, all_relations)

    # list all document ids
    doc_ids = [ doc_id  for doc_id in all_words if doc_id in all_relations ]

    return doc_ids, all_words, all_relations


def build_word2id(all_words, max_size=None, min_count=2, word2id=None):
    """Build vocabulary index for all words (id 0 reserved for masking, id 1 for unknown words)."""
    if word2id is None:
        word2id = {}

    # count word occurrences
    vocab_cnts = {}
    for doc_id in all_words:
        for word in all_words[doc_id]:
            try:
                vocab_cnts[word['Text']] += 1
            except KeyError:
                vocab_cnts[word['Text']] = 1

    # ignore words with low occurrences
    vocab_cnts = dict([ (w, cnt)  for w, cnt in vocab_cnts.iteritems() if cnt >= min_count ])

    # rank words by decreasing occurrences and use as index
    word2id_rev = [None, ""] + sorted(vocab_cnts, key=vocab_cnts.get, reverse=True)
    if max_size is not None:
        word2id_rev = word2id_rev[:max_size]

    # mapping of words to vocabulary ids
    word2id.update([ (w, i) for i, w in enumerate(word2id_rev) ])
    return word2id


def build_pos2id(all_words, max_size=None, min_count=0, pos2id=None):
    """Build POS tags index for all words (id 0 reserved for masking, id 1 for unknown POS tags)."""
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


def build_pdtbpair2id():
    """Build PDTB-style discourse parsing span pairs index and weights."""

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


### Build numpy arrays

def conv_window_to_offsets(window_size, negative_samples, word_crop):
    """Convert window size and negative samples to list of offsets."""

    offsets = range(-window_size // 2, window_size // 2 + 1)
    if window_size % 2 == 0:
        del offsets[window_size // 2]
    for i in range(negative_samples):
        offsets.append(word_crop + i)
    return offsets


def build_x_word(doc_ids, all_words, word2id, word_crop, max_len):
    """Prepare input: word ids with masked and random post-padding (doc, time_pad)."""

    x_word_pad = []
    x_word_rand = []
    for doc_id in doc_ids:
        # map words to vocabulary ids
        ids = []
        for word in all_words[doc_id][:word_crop]:
            try:
                ids.append(word2id[word['Text']])
            except KeyError:  # missing in vocabulary
                ids.append(word2id[""])

        # convert to numpy array with masked and random post-padding
        x_word_pad.append(np.hstack([ids, np.zeros((max_len - len(ids),), dtype=np.int)]))
        x_word_rand.append(np.hstack([ids, np.random.randint(1, len(word2id), size=max_len - len(ids))]))

    # return as numpy array
    x_word_pad = np.asarray(x_word_pad)
    x_word_rand = np.asarray(x_word_rand)
    return x_word_pad, x_word_rand


def build_y_skipgram(x_word_pad, skipgram_offsets, max_len):
    """Prepare output: skip-gram labels (doc, time_pad, offset)."""

    y_skipgram = []
    for s in x_word_pad:
        # map word pairs with mask to binary skip-gram labels
        pairs = [ [ (s[i] != 0 and s[(i + off) % max_len] != 0)  for off in skipgram_offsets ]  for i in range(max_len) ]
        y_skipgram.append(np.asarray(pairs))

    # return as numpy array
    y_skipgram = np.asarray(y_skipgram)
    return y_skipgram


def build_y_pos(doc_ids, all_words, pos2id, word_crop, max_len):
    """Prepare output: POS tags (doc, time_pad, pos2id)."""

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


def build_y_pdtbpair(doc_ids, all_words, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, word_crop, max_len, filter_prefixes=None):
    """Prepare output: PDTB-style discourse relation pairwise occurrences (doc, time, offset, pdtbpair2id)."""

    y_pdtbpair = []
    for doc_id in doc_ids:
        doc_len = len(all_words[doc_id])

        # map word pairs with PDTB-style tags to pairwise occurrences
        pairs = np.zeros((max_len, len(pdtbpair_offsets), pdtbpair2id_size))
        for w1_i in range(max_len):  # iterate word 1

            # filtered word 1 tags by specified relation tags
            w1_tags = []
            if w1_i < word_crop and w1_i < doc_len:
                w1_tags = all_words[doc_id][w1_i]['Tags'].keys()
                w1_tags = conll15st_relations.filter_tags(w1_tags, filter_prefixes)

            for off_i, off in enumerate(pdtbpair_offsets):  # iterate word 2
                w2_i = (w1_i + off) % max_len
                pdtbpair2id_key = None

                # filtered word 2 tags by specified relation tags
                w2_tags = []
                if w2_i < word_crop and w2_i < doc_len:
                    w2_tags = all_words[doc_id][w2_i]['Tags'].keys()
                    w2_tags = conll15st_relations.filter_tags(w2_tags, filter_prefixes)

                # mark occurrences with word 1 and word 2
                for w1_tag in w1_tags:
                    w1_rtype, w1_rsense, w1_rnum, w1_rspan = conll15st_relations.tag_to_rtsns(w1_tag)

                    # check if word 2 is in same relation
                    pdtbpair2id_key = "{}-Rest".format(w1_rspan)
                    for w2_tag in w2_tags:
                        w2_rtype, w2_rsense, w2_rnum, w2_rspan = conll15st_relations.tag_to_rtsns(w2_tag)

                        if w1_rtype == w2_rtype and w1_rsense == w2_rsense and w1_rnum == w2_rnum:
                            pdtbpair2id_key = "{}-{}".format(w1_rspan, w2_rspan)
                            break

                    # update pair
                    pairs[w1_i, off_i, pdtbpair2id[pdtbpair2id_key]] += pdtbpair2id_weights[pdtbpair2id_key]

                # else mark occurrences with only word 2
                if not w1_tags:
                    for w2_tag in w2_tags:
                        w2_rtype, w2_rsense, w2_rnum, w2_rspan = conll15st_relations.tag_to_rtsns(w2_tag)

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


### Interpret numpy arrays

def fitness_partial(pairs, offsets, pair2id, pair2id_weights, sets, update_sets):
    """Evaluate fitness difference after applying updates to pairwise occurrences."""

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


def extract_max_spans(pairs, offsets, pair2id, pair2id_weights, sets, sets_max_len):
    """Extract max spans from pairwise occurrences."""

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
                fitness = fitness_partial(pairs, offsets, pair2id, pair2id_weights, sets, update_sets)
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


def interpret_y_pdtbpair(doc_ids, all_words, y_pdtbpair, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, max_len, rtype, rsense):
    """Interpret pairwise occurrences as PDTB-style discourse relations."""

    sets_max_len = {  #XXX
        'Arg1': 30,
        'Arg2': 30,
        'Connective': 5,
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
            sets = extract_max_spans(y_pdtbpair[d], pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, sets, sets_max_len)

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


### Main

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
    argp.add_argument('experiment_dir',
        help="directory for storing trained model and other resources")
    argp.add_argument('train_dir',
        help="CoNLL15st dataset directory for training")
    argp.add_argument('valid_dir',
        help="CoNLL15st dataset directory for validation")
    argp.add_argument('test_dir',
        help="CoNLL15st dataset directory for testing (only 'pdtb-parses.json')")
    argp.add_argument('output_dir',
        help="output directory for system predictions (in 'output.json')")
    argp.add_argument('--clean', action='store_true',
        help="clean previous experiment")
    args = argp.parse_args()

    # defaults
    epochs = 1000

    word_crop = 1000  #= max([ len(s)  for s in train_words ])
    embedding_dim = 40
    word2id_size = 50000  #= None is computed
    skipgram_window_size = 4
    skipgram_negative_samples = 0  #skipgram_window_size
    skipgram_offsets = conv_window_to_offsets(skipgram_window_size, skipgram_negative_samples, word_crop)
    pos2id_size = 20  #= None is computed
    pdtbpair2id_size = 16  #=16 is fixed
    pdtbpair_window_size = 20  #20
    pdtbpair_negative_samples = 0  #1
    pdtbpair_offsets = conv_window_to_offsets(pdtbpair_window_size, pdtbpair_negative_samples, word_crop)
    filter_prefixes = ["Explicit:Expansion.Conjunction:1"]
    rtype = filter_prefixes[0].split(":")[0]
    rsense = filter_prefixes[0].split(":")[1]
    max_len = word_crop + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)), abs(min(pdtbpair_offsets)), abs(max(pdtbpair_offsets)))

    log.info("configuration ({})".format(args.experiment_dir))
    for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'args.test_dir', 'args.output_dir', 'word_crop', 'embedding_dim', 'word2id_size', 'skipgram_window_size', 'skipgram_negative_samples', 'skipgram_offsets', 'pos2id_size', 'pdtbpair2id_size', 'pdtbpair_window_size', 'pdtbpair_negative_samples', 'pdtbpair_offsets', 'filter_prefixes', 'max_len']:
        log.info("  {}: {}".format(var, eval(var)))

    # experiment files
    if args.clean and os.path.isdir(args.experiment_dir):
        import shutil
        shutil.rmtree(args.experiment_dir)
    if not os.path.isdir(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    word2id_pkl = "{}/word2id.pkl".format(args.experiment_dir)
    pos2id_pkl = "{}/pos2id.pkl".format(args.experiment_dir)
    pdtbpair2id_pkl = "{}/pdtbpair2id.pkl".format(args.experiment_dir)
    model_yaml = "{}/model.yaml".format(args.experiment_dir)
    model_png = "{}/model.png".format(args.experiment_dir)
    stats_csv = "{}/stats.csv".format(args.experiment_dir)
    weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    word2vec_dim = 300

    # load datasets
    log.info("load dataset for training ({})".format(args.train_dir))
    train_doc_ids, train_words, train_relations = load_conll15st(args.train_dir, filter_prefixes=filter_prefixes)
    log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(train_doc_ids), sum([ len(s)  for s in train_words.itervalues() ]), sum([ len(s)  for s in train_relations.itervalues() ])))
    import copy
    train_relations_list = [ r  for doc_id in train_doc_ids for r in copy.deepcopy(train_relations)[doc_id] ]
    for r in train_relations_list:
        r['Arg1']['TokenList'] = [ [0, 0, i, 0, 0]  for i in r['Arg1']['TokenList'] ]
        r['Arg2']['TokenList'] = [ [0, 0, i, 0, 0]  for i in r['Arg2']['TokenList'] ]
        r['Connective']['TokenList'] = [ [0, 0, i, 0, 0]  for i in r['Connective']['TokenList'] ]

    #log.info("load dataset for validation ({})".format(args.valid_dir))
    #valid_doc_ids, valid_words, valid_relations = load_conll15st(args.valid_dir, filter_prefixes=filter_prefixes)
    #log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(valid_doc_ids), sum([ len(s)  for s in valid_words.itervalues() ]), sum([ len(s)  for s in valid_relations.itervalues() ])))

    #log.info("load dataset for testing ({})".format(args.test_dir))
    #test_doc_ids, test_words, test_relations = load_conll15st(args.test_dir, filter_prefixes=filter_prefixes)
    #log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(test_doc_ids), sum([ len(s)  for s in test_words.itervalues() ]), sum([ len(s)  for s in test_relations.itervalues() ])))

    # build indexes
    if not os.path.isfile(word2id_pkl) or not os.path.isfile(pos2id_pkl) or not os.path.isfile(pdtbpair2id_pkl):
        log.info("build indexes")
        word2id = build_word2id(train_words, max_size=word2id_size)
        pos2id = build_pos2id(train_words, max_size=pos2id_size)
        pdtbpair2id, pdtbpair2id_weights = build_pdtbpair2id()
        with open(word2id_pkl, 'wb') as f:
            pickle.dump(word2id, f)
        with open(pos2id_pkl, 'wb') as f:
            pickle.dump(pos2id, f)
        with open(pdtbpair2id_pkl, 'wb') as f:
            pickle.dump((pdtbpair2id, pdtbpair2id_weights), f)
    else:
        log.info("load previous indexes ({})".format(args.experiment_dir))
        with open(word2id_pkl, 'rb') as f:
            word2id = pickle.load(f)
        with open(pos2id_pkl, 'rb') as f:
            pos2id = pickle.load(f)
        with open(pdtbpair2id_pkl, 'rb') as f:
            pdtbpair2id, pdtbpair2id_weights = pickle.load(f)
    log.info("  word2id: {}, pos2id: {}, pdtbpair2id: {}".format(len(word2id), len(pos2id), len(pdtbpair2id)))

    # build model
    log.info("build model")
    model = arch.build(max_len, embedding_dim, len(word2id), skipgram_offsets, len(pos2id), len(pdtbpair2id), pdtbpair_offsets)

    # plot model
    with open(model_yaml, 'w') as f:
        model.to_yaml(stream=f)
    plot(model, model_png)

    # initialize model
    if not os.path.isfile(weights_hdf5):
        log.info("initialize new model")
        #XXX: arch.init_word2vec(model, word2vec_bin, word2vec_dim)
    else:
        log.info("load previous model ({})".format(args.experiment_dir))
        model.load_weights(weights_hdf5)

    # initialize performance stats
    stats = Stats(experiment=args.experiment_dir, train_dir=args.train_dir, valid_dir=args.valid_dir)
    if not os.path.isfile(stats_csv):
        log.info("initialize stats")
        stats.save(stats_csv)
    else:
        log.info("load previous stats ({})".format(args.experiment_dir))
        stats.load(stats_csv)

    epoch_i = -1
    loss_best = float('inf')
    if stats.history:
        epoch_i = int(stats.history[-1]['epoch'])
        loss_avg = float(stats.history[-1]['loss_avg'])
        loss_best = loss_avg
        loss_min = float(stats.history[-1]['loss_min'])
        loss_max = float(stats.history[-1]['loss_max'])
        log.info("  continue from epoch {}, loss avg: {:.4f}, min: {:.4f}, max: {:.4f}".format(epoch_i, loss_avg, loss_min, loss_max))

    # train model
    while epoch_i < epochs:
        epoch_i += 1
        time_0 = time.time()
        log.info("train epoch {}/{} ({} docs)".format(epoch_i, epochs, len(train_doc_ids)))

        # one document per batch update
        loss_avg = 0.
        loss_min = np.inf
        loss_max = -np.inf
        for batch_i, doc_id in enumerate(train_doc_ids):
            # prepare batch data
            doc_ids = [doc_id]
            from pprint import pprint

            x_word_pad, x_word_rand = build_x_word(doc_ids, train_words, word2id, word_crop, max_len)
            #print("x_word_pad:"); pprint(x_word_pad)
            #print("x_word_rand:"); pprint(x_word_rand)

            y_skipgram = build_y_skipgram(x_word_pad, skipgram_offsets, max_len)
            #print("y_skipgram:"); pprint(y_skipgram)

            y_pos = build_y_pos(doc_ids, train_words, pos2id, word_crop, max_len)
            #print("y_pos:"); pprint(y_pos)

            y_pdtbpair = build_y_pdtbpair(doc_ids, train_words, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, word_crop, max_len)
            #print("y_pdtbpair:"); pprint(y_pdtbpair)

            # train on batch
            loss = model.train_on_batch({
                'x_word_pad': x_word_pad,
                'x_word_rand': x_word_rand,
                'y_skipgram': y_skipgram,
                'y_pos': y_pos,
                'y_pdtbpair': y_pdtbpair,
            })

            #XXX
            aa = {
                'x_word_pad': x_word_pad,
                'x_word_rand': x_word_rand,
                'y_skipgram': y_skipgram,
                'y_pos': y_pos,
                'y_pdtbpair': y_pdtbpair,
            }
            # print "layer_1"
            # layer_1 = arch.get_activations(model, 'layer_1', aa)
            # pprint(layer_1[0].shape)
            # pprint(layer_1[0])
            # print "layer_2"
            # layer_2 = arch.get_activations(model, 'layer_2', aa)
            # pprint(layer_2[0].shape)
            # pprint(layer_2[0])
            # print "pdtbpair_offsets"
            # pdtbpair_offsets = arch.get_activations(model, 'pdtbpair_offsets', aa)
            # pprint(pdtbpair_offsets[0].shape)
            # pprint(pdtbpair_offsets[0])
            # print "pdtbpair_repeat"
            # pdtbpair_repeat = arch.get_activations(model, 'pdtbpair_repeat', aa)
            # pprint(pdtbpair_repeat[0].shape)
            # pprint(pdtbpair_repeat[0])
            # print "pdtbpair_dense"
            # pdtbpair_dense = arch.get_activations(model, 'pdtbpair_dense', aa)
            # pprint(pdtbpair_dense[0].shape)
            # pprint(pdtbpair_dense[0])
            # raise Exception()
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = model.train_on_batch(aa)
            loss = float(loss)

            # compute stats
            loss_avg += loss
            if loss < loss_min:
                loss_min = loss
            if loss > loss_max:
                loss_max = loss

        loss_avg /= len(train_doc_ids)
        time_1 = time.time()
        log.info("  loss avg: {:.8f}, min: {:.8f}, max: {:.8f}, time: {:.1f}s".format(loss_avg, loss_min, loss_max, time_1 - time_0))

        # validate model on training dataset
        relations_list = []
        for batch_i, doc_id in enumerate(train_doc_ids):

            # prepare batch data
            doc_ids = [doc_id]
            x_word_pad, x_word_rand = build_x_word(doc_ids, train_words, word2id, word_crop, max_len)

            # interpret predictions as relations
            y = model.predict({
                'x_word_pad': x_word_pad,
                'x_word_rand': x_word_rand,
            })
            all_relations = interpret_y_pdtbpair(doc_ids, train_words, y['y_pdtbpair'], pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, max_len, rtype, rsense)
            relations_list.extend([ r  for doc_id in doc_ids for r in all_relations[doc_id] ])

        # evaluate relations on training dataset
        train_precision, train_recall, train_f1 = scorer.evaluate_relation(train_relations_list, relations_list)
        time_2 = time.time()
        log.info("  train precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, relations: {}, time: {:.1f}s".format(train_precision, train_recall, train_f1, len(relations_list), time_2 - time_1))

        #XXX
        valid_precision = valid_recall = valid_f1 = -1
        time_3 = time.time()

        # save stats
        stats.append({
            'epoch': epoch_i,
            'loss_avg': loss_avg,
            'loss_min': loss_min,
            'loss_max': loss_max,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
            'valid_f1': valid_f1,
            'time_1': time_1 - time_0,
            'time_2': time_2 - time_1,
            'time_3': time_3 - time_2,
        })
        stats.save(stats_csv)

        # save best model
        if loss_avg < loss_best:
            loss_best = loss_avg
            model.save_weights(weights_hdf5, overwrite=True)

    # predict model
    # log.info("predict model")

    # batches_len = len(test_doc_ids)
    # for batch_i, doc_id in enumerate(test_doc_ids):
    #     log.info("predict batch {}/{}".format(batch_i, batches_len))

    #     # prepare batch data
    #     data = {} #TODO
    #     predictions = model.predict(data)

    #     # store batch
    #     #TODO output_dir/output.json
