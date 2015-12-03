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
        self.fieldnames = ['experiment', 'train_dir', 'valid_dir', 'epoch', 'loss_avg', 'loss_min', 'loss_max', 'train_precision', 'train_recall', 'train_f1', 'valid_precision', 'valid_recall', 'valid_f1', 'epoch_time']

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

def load_conll15st(dataset_dir):
    """Load CoNLL15st dataset in original form."""

    # load all relations by document id
    all_relations = conll15st_relations.load_relations(dataset_dir)
    all_relations = conll15st_relations.conv_span_tokenlist_format(all_relations)
    all_relations = conll15st_relations.add_relation_sensenum(all_relations)

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
    """Build PDTB-style discourse parsing span pairs index."""

    pdtbpair2id = {
        "Rest-Rest": 0,
        "Arg1-Arg1": 1,
        "Arg1-Arg2": 2,
        "Arg1-Connective": 3,
        "Arg1-Rest": 4,
        "Arg2-Arg1": 5,
        "Arg2-Arg2": 6,
        "Arg2-Connective": 7,
        "Arg2-Rest": 8,
        "Connective-Arg1": 9,
        "Connective-Arg2": 10,
        "Connective-Connective": 11,
        "Connective-Rest": 12,
        "Rest-Arg1": 13,
        "Rest-Arg2": 14,
        "Rest-Connective": 15,
    }
    return pdtbpair2id


### Prepare NumPy arrays

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


def build_y_pdtbpair(doc_ids, all_words, pdtbpair_offsets, pdtbpair2id, word_crop, max_len, filter_prefixes):
    """Prepare output: PDTB-style discourse relation span pairs (doc, time, offset, pdtbpair2id)."""

    y_pdtbpair = []
    for doc_id in doc_ids:
        doc_len = len(all_words[doc_id])

        # map word pairs with tags to PDTB-style span pair occurrences
        pairs = np.zeros((max_len, len(pdtbpair_offsets), pdtbpair2id_size), dtype=np.int)
        for i in range(max_len):  # iterate word 1

            # filtered word 1 tags by specified relation tags
            w1_tags = []
            if i < word_crop and i < doc_len:
                w1_tags = all_words[doc_id][i]['Tags'].keys()
                w1_tags = conll15st_relations.filter_tags(w1_tags, filter_prefixes)

            for off_i, off in enumerate(pdtbpair_offsets):  # iterate word 2
                j = (i + off) % max_len
                pdtbpair2id_key = None

                # filtered word 2 tags by specified relation tags
                w2_tags = []
                if j < word_crop and j < doc_len:
                    w2_tags = all_words[doc_id][j]['Tags'].keys()
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
                    pairs[i, off_i, pdtbpair2id[pdtbpair2id_key]] += 1

                # else mark occurrences with only word 2
                if not w1_tags:
                    for w2_tag in w2_tags:
                        w2_rtype, w2_rsense, w2_rnum, w2_rspan = conll15st_relations.tag_to_rtsns(w2_tag)

                        # no word 1 tags
                        pdtbpair2id_key = "Rest-{}".format(w2_rspan)

                        # update pair
                        pairs[i, off_i, pdtbpair2id[pdtbpair2id_key]] += 1

                # else mark no occurrences between word 1 and word 2
                if pdtbpair2id_key is None:
                    pdtbpair2id_key = "Rest-Rest"

                    # update pair
                    pairs[i, off_i, pdtbpair2id[pdtbpair2id_key]] += 1
        y_pdtbpair.append(pairs)

    # return as numpy array
    y_pdtbpair = np.asarray(y_pdtbpair)
    return y_pdtbpair


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
    args = argp.parse_args()

    # defaults
    epochs = 1000

    word_crop = 1000  #= max([ len(s)  for s in train_words ])
    embedding_dim = 3
    word2id_size = 50000  #= None is computed
    skipgram_window_size = 4
    skipgram_negative_samples = 1  #skipgram_window_size
    skipgram_offsets = conv_window_to_offsets(skipgram_window_size, skipgram_negative_samples, word_crop)
    pos2id_size = 20  #= None is computed
    pdtbpair2id_size = 16  #=16 is fixed
    pdtbpair_window_size = 4  #20
    pdtbpair_negative_samples = 0  #1
    pdtbpair_offsets = conv_window_to_offsets(pdtbpair_window_size, pdtbpair_negative_samples, word_crop)
    pdtbpair_filter_prefixes = ["Explicit:Expansion.Conjunction"]
    max_len = word_crop + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)), abs(min(pdtbpair_offsets)), abs(max(pdtbpair_offsets)))

    log.info("configuration ({})".format(args.experiment_dir))
    for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'args.test_dir', 'args.output_dir', 'word_crop', 'embedding_dim', 'word2id_size', 'skipgram_window_size', 'skipgram_negative_samples', 'skipgram_offsets', 'pos2id_size', 'pdtbpair2id_size', 'pdtbpair_window_size', 'pdtbpair_negative_samples', 'pdtbpair_offsets', 'pdtbpair_filter_prefixes', 'max_len']:
        log.info("  {}: {}".format(var, eval(var)))

    # experiment files
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
    train_doc_ids, train_words, train_relations = load_conll15st(args.train_dir)
    log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(train_doc_ids), sum([ len(s)  for s in train_words.itervalues() ]), sum([ len(s)  for s in train_relations.itervalues() ])))

    #log.info("load dataset for validation ({})".format(args.valid_dir))
    #valid_doc_ids, valid_words, valid_relations = load_conll15st(args.valid_dir)
    #log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(valid_doc_ids), sum([ len(s)  for s in valid_words.itervalues() ]), sum([ len(s)  for s in valid_relations.itervalues() ])))

    #log.info("load dataset for testing ({})".format(args.test_dir))
    #test_doc_ids, test_words, test_relations = load_conll15st(args.test_dir)
    #log.info("  doc_ids: {}, all_words: {}, all_relations: {}".format(len(test_doc_ids), sum([ len(s)  for s in test_words.itervalues() ]), sum([ len(s)  for s in test_relations.itervalues() ])))

    # build indexes
    if not os.path.isfile(word2id_pkl) or not os.path.isfile(pos2id_pkl) or not os.path.isfile(pdtbpair2id_pkl):
        log.info("build indexes")
        word2id = build_word2id(train_words, max_size=word2id_size)
        pos2id = build_pos2id(train_words, max_size=pos2id_size)
        pdtbpair2id = build_pdtbpair2id()
        with open(word2id_pkl, 'wb') as f:
            pickle.dump(word2id, f)
        with open(pos2id_pkl, 'wb') as f:
            pickle.dump(pos2id, f)
        with open(pdtbpair2id_pkl, 'wb') as f:
            pickle.dump(pdtbpair2id, f)
    else:
        log.info("load previous indexes ({})".format(args.experiment_dir))
        with open(word2id_pkl, 'rb') as f:
            word2id = pickle.load(f)
        with open(pos2id_pkl, 'rb') as f:
            pos2id = pickle.load(f)
        with open(pdtbpair2id_pkl, 'rb') as f:
            pdtbpair2id = pickle.load(f)
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
        epoch_time_0 = time.time()
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

            y_pdtbpair = build_y_pdtbpair(doc_ids, train_words, pdtbpair_offsets, pdtbpair2id, word_crop, max_len, pdtbpair_filter_prefixes)
            #print("y_pdtbpair:"); pprint(y_pdtbpair)

            # train on batch
            loss = model.train_on_batch({
                'x_word_pad': x_word_pad,
                'x_word_rand': x_word_rand,
                'y_skipgram': y_skipgram,
                'y_pos': y_pos,
                'y_pdtbpair': y_pdtbpair,
            })
            loss = float(loss)

            # compute stats
            loss_avg += loss
            if loss < loss_min:
                loss_min = loss
            if loss > loss_max:
                loss_max = loss

        loss_avg /= len(train_doc_ids)
        log.info("  loss avg: {:.4f}, min: {:.4f}, max: {:.4f}".format(loss_avg, loss_min, loss_max))

        # validate model
        ###TODO
        train_precision = -1.0
        train_recall = -1.0
        train_f1 = -1.0
        valid_precision = -1.0
        valid_recall = -1.0
        valid_f1 = -1.0

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
            'epoch_time': time.time() - epoch_time_0,
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
