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

def load(dataset_dir, word2vec_bin, word2vec_dim, tag_to_j):
    """Load PDTB data and transform it to numerical form."""



    # load relations by document id
    relations = conll15st_relations.load_relations(dataset_dir)

    # load words by document id
    words = conll15st_words.load_words(dataset_dir)
    words = conll15st_relations.conv_linkers_to_tags(words, relations)



    # prepare mapping vocabulary to word2vec vectors
    map_word2vec = joblib.load("./ex02_model/map_word2vec.dump")  #XXX

    # load words from PDTB parses
    words = load_words(pdtb_dir, relations)

    # prepare numeric form
    x = []
    y = []
    doc_ids = []
    for doc_id, doc in words.iteritems():
        doc_x = []
        doc_y = []
        for word in doc:
            # map text to word2vec
            try:
                doc_x.append(map_word2vec[word['Text']])
            except KeyError:  # missing in vocab
                doc_x.append(np.zeros(word2vec_dim))

            # map tags to vector
            tags = [0.] * len(tag_to_j)
            for tag, count in word['Tags'].iteritems():
                tags[tag_to_j[tag]] = float(count)
            doc_y.append(tags)

            #print word['Text'], word['Tags']
            #print word['Text'], doc_x[-1][0:1], doc_y[-1]

        x.append(np.asarray(doc_x, dtype=theano.config.floatX))
        y.append(np.asarray(doc_y, dtype=theano.config.floatX))
        doc_ids.append(doc_id)
        if doc_id not in relations:
            relations[doc_id] = []

    return x, y, doc_ids, words, relations


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
    stats_csv = "{}/stats.csv".format(args.experiment_dir)
    weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    word2vec_dim = 300
    vocabulary_n = 10000

    epochs = 10

    # build model
    log.info("build model")
    stats = Stats(experiment=args.experiment_dir, train_dir=args.train_dir, valid_dir=args.valid_dir)
    model = arch.build(vocabulary_n, word2vec_dim)

    model.get_config(verbose=1)
    from keras.utils.dot_utils import Grapher
    grapher = Grapher()
    grapher.plot(model, "{}/model.png".format(args.experiment_dir))

    # initialize model
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
