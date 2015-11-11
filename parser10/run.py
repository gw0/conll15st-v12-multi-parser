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
import numpy as np
import theano
import theano.tensor as T

import arch
import conll15st_relations
import conll15st_words


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
        self.fieldnames = ["experiment", "train_dir", "valid_dir", "epoch", "learn_rate", "train_precision", "train_recall", "train_f1", "valid_precision", "valid_recall", "valid_f1", "cost_min", "cost_max", "cost_avg", "y_min_avg", "y_max_avg", "y_mean_avg", "epoch_time"]

        self.experiment = experiment
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.history = []

    def append(self, row):
        row["experiment"] = self.experiment
        row["train_dir"] = self.train_dir
        row["valid_dir"] = self.valid_dir
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

def build_word2id(words_all, max_vocab_size=None, min_count=2, word2id=None):
    """Build vocabulary index for all words."""
    if word2id is None:
        word2id = {}

    # count word occurrences
    vocab_cnts = {}
    for doc_id in words_all:
        for word in words_all[doc_id]:
            try:
                vocab_cnts[word['Text']] += 1
            except KeyError:
                vocab_cnts[word['Text']] = 1

    # ignore words with low occurrences
    vocab_cnts = dict([ (w, cnt)  for w, cnt in vocab_cnts.iteritems() if cnt >= min_count ])

    # rank words by decreasing occurrences and use as index
    word2id_rev = [''] + sorted(vocab_cnts, key=vocab_cnts.get, reverse=True)
    if max_vocab_size is not None:
        word2id_rev = word2id_rev[:max_vocab_size]

    # mapping of words to vocabulary ids
    word2id.update([ (w, i) for i, w in enumerate(word2id_rev) ])
    return word2id


def build_pos2id(words_all, max_pos_size=None, min_count=0, pos2id=None):
    """Build POS tags index for all words."""
    if pos2id is None:
        pos2id = {}

    # count POS tags occurrences
    pos_cnts = {}
    for doc_id in words_all:
        for word in words_all[doc_id]:
            try:
                pos_cnts[word['PartOfSpeech']] += 1
            except KeyError:
                pos_cnts[word['PartOfSpeech']] = 1

    # ignore POS tags with low occurrences
    for w, cnt in pos_cnts.iteritems():
        if cnt < min_count:
            del pos_cnts[w]

    # rank POS tags by decreasing occurrences and use as index
    pos2id_rev = [''] + sorted(pos_cnts, key=pos_cnts.get, reverse=True)
    if max_pos_size is not None:
        pos2id_rev = pos2id_rev[:max_pos_size]

    # mapping of POS tags to ids
    pos2id.update([ (w, i) for i, w in enumerate(pos2id_rev) ])
    return pos2id


def build_x_word(doc_ids, words_all, word2id):
    """Prepare numpy array for x_word (doc, time, word id)."""

    x_word = []
    for doc_id in doc_ids:
        word_ids = []
        for word in words_all[doc_id]:
            # map words to vocabulary ids
            try:
                word_ids.append(word2id[word['Text']])
            except KeyError:  # missing in vocabulary
                word_ids.append(word2id[''])

        # store as numpy array
        x_word.append(np.asarray(word_ids, dtype=theano.config.floatX))
    return x_word


def gen_pairs_window(seq1, seq2=None, window_size=4, window_offsets=None, padding=None):
    """Generate seq1-seq2 pairs where seq2 is within sliding window of seq1 element."""
    if seq2 is None:
        seq2 = seq1
    if window_offsets is None:
        window_offsets = range(-window_size // 2, window_size // 2 + 1)
        if window_size % 2 == 0:
            del window_offsets[window_size // 2]

    # padding for seq2
    window_min = min(window_offsets)
    window_max = max(window_offsets)
    seq2 = [padding] * -window_min + seq1 + [padding] * window_max

    # generate seq1-seq2 pairs in sliding window
    pairs_off = [ zip(seq1, seq2[off - window_min:])  for off in window_offsets ]
    pairs = map(list, zip(*pairs_off))
    return pairs


def build_y_skipgram(doc_ids, words_all, window_size):
    """Prepare numpy array for y_skipgram (doc, time, window, SG label)."""

    y_skipgram = []
    for doc_id in doc_ids:
        # fill word-context pair labels for skip-gram model
        pairs = gen_pairs_window(words_all[doc_id], window_size=window_size)
        for i in range(len(pairs)):
            for j in range(len(pairs[0])):
                pairs[i][j] = 1

        # store as numpy array
        y_skipgram.append(np.asarray(pairs, dtype=theano.config.floatX))
    return y_skipgram


def build_y_pos(doc_ids, words_all, pos2id):
    """Prepare numpy array for y_pos (doc, time, POS tag)."""

    y_pos = []
    for doc_id in doc_ids:
        doc_pos = []
        for word in words_all[doc_id]:
            # map POS tags to ids
            try:
                doc_pos.append(pos2id[word['PartOfSpeech']])
            except KeyError:  # missing in index
                doc_pos.append(pos2id[''])

        # store as numpy array
        y_pos.append(np.asarray(doc_pos, dtype=theano.config.floatX))
    return y_pos


def load(dataset_dir, vocab_size=None, skipgram_window_size=4):
    """Load CoNLL15st dataset and transform it to numerical form."""

    # load all relations by document id
    relations_all = conll15st_relations.load_relations_all(dataset_dir)
    relations_all = conll15st_relations.conv_tokenlists(relations_all)
    relations_all = conll15st_relations.conv_sensenum(relations_all)

    # load all words by document id
    words_all = conll15st_words.load_words_all(dataset_dir)
    words_all = conll15st_relations.conv_linkers_to_tags(words_all, relations_all)

    # build word vocabulary index
    word2id = build_word2id(words_all, max_vocab_size=vocab_size)

    # build POS tags index
    pos2id = build_pos2id(words_all)

    # fixate order of document ids
    doc_ids = [ doc_id  for doc_id in words_all if doc_id in relations_all ]

    # word ids in numpy (doc, time, word id)
    x_word = build_x_word(doc_ids, words_all, word2id)

    # skip-gram model word-context pair labels in numpy (doc, time, window, SG label)
    y_skipgram = build_y_skipgram(doc_ids, words_all, window_size=skipgram_window_size)

    # POS tag ids of words in numpy (doc, time, POS tag)
    y_pos = build_y_pos(doc_ids, words_all, pos2id)

    # discourse relations word-word pair labels per relation sense in numpy ([sense], doc, time, window, relation label)
    y_relation = []  #TODO

    return x_word, y_skipgram, y_pos, y_relation, doc_ids, words_all, relations_all, word2id, pos2id


### Main

if __name__ == '__main__':
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
    args = argp.parse_args()

    # defaults
    vocab_size = 10000
    skipgram_window_size = 4

    epochs = 10

    stats_csv = "{}/stats.csv".format(args.experiment_dir)
    weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    word2vec_dim = 300

    # load datasets
    log.info("load datasets")
    x_word, y_skipgram, y_pos, y_relation, doc_ids, words_all, relations_all, word2id, pos2id = load(args.train_dir, vocab_size=vocab_size, skipgram_window_size=skipgram_window_size)
    rand_size = max([ len(word_ids) for word_ids in x_word ])
    x_rand = np.random.randint(low=1, high=vocab_size, size=(rand_size,))

    print "x_word:", x_word[0].shape, sum([ x.nbytes  for x in x_word ])
    print "vocab_size:", len(word2id)
    print "y_skipgram:", y_skipgram[0].shape, sum([ y.nbytes  for y in y_skipgram ])
    print "y_pos:", y_pos[0].shape, sum([ y.nbytes  for y in y_pos ])
    print "pos_size:", len(pos2id)
    print "y_relation:", y_relation
    exit()    #TODO

    # build model
    log.info("build model")
    model = arch.build(vocab_size, word2vec_dim)

    model.get_config(verbose=1)
    from keras.utils.dot_utils import Grapher
    grapher = Grapher()
    grapher.plot(model, "{}/model.png".format(args.experiment_dir))

    # initialize model
    stats = Stats(experiment=args.experiment_dir, train_dir=args.train_dir, valid_dir=args.valid_dir)
    if not os.path.isdir(args.experiment_dir):
        log.info("initialize new model")
        stats.save(stats_csv)
        arch.init_word2vec(model, word2vec_bin, word2vec_dim)
    else:
        log.info("load previous model")
        stats.load(stats_csv)
        model.load_weights(weights_hdf5)

    # train model
    epoch = 0
    while epoch < epochs:
        log.info("train epoch {}".format(epoch))

        #keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_best_only=False)
        model.fit()  #TODO
        #stats.append({
        #    "epoch": epoch,
        #    "learn_rate": learn_rate,
        #    "train_precision": train_precision,
        #    "train_recall": train_recall,
        #    "train_f1": train_f1,
        #    "valid_precision": valid_precision,
        #    "valid_recall": valid_recall,
        #    "valid_f1": valid_f1,
        #    "cost_min": cost_min,
        #    "cost_max": cost_max,
        #    "cost_avg": cost_avg,
        #    "y_min_avg": y_min_avg,
        #    "y_max_avg": y_max_avg,
        #    "y_mean_avg": y_mean_avg,
        #    "epoch_time": epoch_time,
        #})

        model.save_weights(weights_hdf5, overwrite=True)
        stats.save(stats_csv)

    # predict model
    log.info("predict model")
    model.predict()  #TODO
    stats.save(stats_csv)
