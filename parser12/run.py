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
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.models import make_batches

import conll15st_relations
import conll15st_words
from conll15st_scorer import scorer
from tasks import skipgram, pos, pdtbmark, pdtbpair
import arch


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
        self.log.error("(time {:.2f}s, memory {:+.1f}MB, total {:.3f}GB)".format(self.time_1 - self.time_0, (self.mem_1 - self.mem_0) / 1024.0, self.mem_1 / 1024.0 / 1024.0))

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


def build_word2id(all_words, max_size=None, min_count=1, word2id=None):
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
    epochs = 10000
    batch_size = 10

    word_crop = 1000  #= max([ len(s)  for s in train_words ])
    embedding_dim = 40  #100
    word2id_size = 50000  #= None is computed
    skipgram_window_size = 4
    skipgram_negative_samples = 0  #skipgram_window_size
    skipgram_offsets = conv_window_to_offsets(skipgram_window_size, skipgram_negative_samples, word_crop)
    pos2id_size = 5  #= None is computed
    pdtbmark2id_size = None  #=7 is fixed
    pdtbpair2id_size = None  #=16 is fixed
    pdtbpair_window_size = 20  #40
    pdtbpair_negative_samples = 0  #1
    pdtbpair_offsets = conv_window_to_offsets(pdtbpair_window_size, pdtbpair_negative_samples, word_crop)
    filter_prefixes = ["Explicit:Expansion.Conjunction"]
    rtype = filter_prefixes[0].split(":")[0]
    rsense = filter_prefixes[0].split(":")[1]
    max_len = word_crop + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)), abs(min(pdtbpair_offsets)), abs(max(pdtbpair_offsets)))

    log.info("configuration ({})".format(args.experiment_dir))
    for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'args.test_dir', 'args.output_dir', 'K._config', 'os.getenv("THEANO_FLAGS")', 'epochs', 'batch_size', 'word_crop', 'embedding_dim', 'word2id_size', 'skipgram_window_size', 'skipgram_negative_samples', 'skipgram_offsets', 'pos2id_size', 'pdtbmark2id_size', 'pdtbpair2id_size', 'pdtbpair_window_size', 'pdtbpair_negative_samples', 'pdtbpair_offsets', 'filter_prefixes', 'max_len']:
        log.info("  {}: {}".format(var, eval(var)))

    # experiment files
    if args.clean and os.path.isdir(args.experiment_dir):
        import shutil
        shutil.rmtree(args.experiment_dir)
    if not os.path.isdir(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    word2id_pkl = "{}/word2id.pkl".format(args.experiment_dir)
    pos2id_pkl = "{}/pos2id.pkl".format(args.experiment_dir)
    pdtbmark2id_pkl = "{}/pdtbmark2id.pkl".format(args.experiment_dir)
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
        with open(word2id_pkl, 'wb') as f:
            pickle.dump(word2id, f)
        pos2id = pos.build_pos2id(train_words, max_size=pos2id_size)
        with open(pos2id_pkl, 'wb') as f:
            pickle.dump(pos2id, f)
        pdtbmark2id, pdtbmark2id_size = pdtbmark.build_pdtbmark2id(mode='IO')
        with open(pdtbmark2id_pkl, 'wb') as f:
            pickle.dump((pdtbmark2id, pdtbmark2id_size), f)
        pdtbpair2id, pdtbpair2id_weights = pdtbpair.build_pdtbpair2id()
        with open(pdtbpair2id_pkl, 'wb') as f:
            pickle.dump((pdtbpair2id, pdtbpair2id_weights), f)
    else:
        log.info("load previous indexes ({})".format(args.experiment_dir))
        with open(word2id_pkl, 'rb') as f:
            word2id = pickle.load(f)
        with open(pos2id_pkl, 'rb') as f:
            pos2id = pickle.load(f)
        with open(pdtbmark2id_pkl, 'rb') as f:
            pdtbmark2id, pdtbmark2id_size = pickle.load(f)
        with open(pdtbpair2id_pkl, 'rb') as f:
            pdtbpair2id, pdtbpair2id_weights = pickle.load(f)
    log.info("  word2id: {}, pos2id: {}, pdtbmark2id: {}, pdtbpair2id: {}".format(len(word2id), len(pos2id), pdtbmark2id_size, len(pdtbpair2id)))

    # estimate max achievable results
    relations_list = []
    for batch_i, doc_id in enumerate(train_doc_ids):
        # prepare batch data
        doc_ids = [doc_id]
        y_pdtbpair = pdtbpair.encode_y_pdtbpair(doc_ids, train_words, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, word_crop, max_len)

        # interpret predictions as relations
        all_relations = pdtbpair.decode_y_pdtbpair(doc_ids, train_words, y_pdtbpair, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, max_len, rtype, rsense)
        relations_list.extend([ r  for doc_id in doc_ids for r in all_relations[doc_id] ])

    # evaluate relations on gold dataset
    train_precision, train_recall, train_f1 = scorer.evaluate_relation(train_relations_list, relations_list)
    log.info("  max achievable train precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, relations: {}/{}".format(train_precision, train_recall, train_f1, len(relations_list), len(train_relations_list)))

    # build model
    log.info("build model")
    model = arch.build(max_len, embedding_dim, len(word2id), skipgram_offsets, len(pos2id), pdtbmark2id_size, len(pdtbpair2id), pdtbpair_offsets)

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
        batches = make_batches(len(train_doc_ids), batch_size)
        for batch_i, (batch_start, batch_end) in enumerate(batches):
            # prepare batch data
            doc_ids = train_doc_ids[batch_start:batch_end]

            x_word_pad, x_word_rand = build_x_word(doc_ids, train_words, word2id, word_crop, max_len)
            #print("x_word_pad:", x_word_pad.shape); pprint(x_word_pad)
            #print("x_word_rand:", x_word_rand.shape); pprint(x_word_rand)

            y_skipgram = skipgram.encode_y_skipgram(x_word_pad, skipgram_offsets, max_len)
            #print("y_skipgram:", y_skipgram.shape); pprint(y_skipgram)

            y_pos = pos.encode_y_pos(doc_ids, train_words, pos2id, word_crop, max_len)
            #print("y_pos:", y_pos.shape); pprint(y_pos)

            y_pdtbmark = pdtbmark.encode_y_pdtbmark(doc_ids, train_words, pdtbmark2id, pdtbmark2id_size, word_crop, max_len)
            #print("y_pdtbmark:", y_pdtbmark.shape); pprint(y_pdtbmark)

            y_pdtbpair = pdtbpair.encode_y_pdtbpair(doc_ids, train_words, pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, word_crop, max_len)
            #print("y_pdtbpair:", y_pdtbpair.shape); pprint(y_pdtbpair)

            # train on batch
            loss = model.train_on_batch({
                'x_word_pad': x_word_pad,
                'x_word_rand': x_word_rand,
                'y_skipgram': y_skipgram,
                'y_pos': y_pos,
                'y_pdtbmark': y_pdtbmark,
                'y_pdtbpair': y_pdtbpair,
            })
            loss = float(loss[0])

            # compute stats
            loss_avg += loss
            if loss < loss_min:
                loss_min = loss
            if loss > loss_max:
                loss_max = loss

            #XXX
            try:
                xx_i, xx_f, xx_l = doc_ids.index('wsj_1000'), 408, 428
                #xx_i, xx_f, xx_l = doc_ids.index('wsj_2205'), 461, 480
                y = model.predict({
                    'x_word_pad': x_word_pad,
                    'x_word_rand': x_word_rand,
                })
                np.set_printoptions(precision=2, suppress=True)
                from pprint import pprint
                # {None: 0, '': 1, u'NN': 2, u'NNP': 3, u'IN': 4}
                # print("y_pos:", y_pos.shape); pprint(y_pos[xx_i][xx_f:xx_l])
                # print("y[y_pos]:", y['y_pos'].shape); pprint(y['y_pos'][xx_i][xx_f:xx_l])
                print("y_pdtbmark:", y_pdtbmark.shape); pprint(y_pdtbmark[xx_i][xx_f:xx_l])
                print("y[y_pdtbmark]:", y['y_pdtbmark'].shape); pprint(y['y_pdtbmark'][xx_i][xx_f:xx_l])
            except:
                pass

        loss_avg /= len(train_doc_ids) / float(batch_size)
        time_1 = time.time()
        log.info("  loss avg: {:.2e}, min: {:.2e}, max: {:.2e}, time: {:.2f}s".format(loss_avg, loss_min, loss_max, time_1 - time_0))

        # # validate model on training dataset
        # relations_list = []
        # for batch_i, doc_id in enumerate(train_doc_ids):

        #     # prepare batch data
        #     doc_ids = [doc_id]
        #     x_word_pad, x_word_rand = build_x_word(doc_ids, train_words, word2id, word_crop, max_len)

        #     # interpret predictions as relations
        #     y = model.predict({
        #         'x_word_pad': x_word_pad,
        #         'x_word_rand': x_word_rand,
        #     })
        #     all_relations = pdtbpair.decode_y_pdtbpair(doc_ids, train_words, y['y_pdtbpair'], pdtbpair_offsets, pdtbpair2id, pdtbpair2id_weights, max_len, rtype, rsense)
        #     relations_list.extend([ r  for doc_id in doc_ids for r in all_relations[doc_id] ])

        # # evaluate relations on training dataset
        # train_precision, train_recall, train_f1 = scorer.evaluate_relation(train_relations_list, relations_list)
        # time_2 = time.time()
        # log.info("  train precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, relations: {}/{}, time: {:.2f}s".format(train_precision, train_recall, train_f1, len(relations_list), len(train_relations_list), time_2 - time_1))
        # if len(relations_list) > 0 and (train_precision > 0. or train_recall > 0. or train_f1 > 0.):
        #     print "  WOOHOOO!!!"

        #XXX
        train_precision = train_recall = train_f1 = -1
        time_2 = time.time()
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
