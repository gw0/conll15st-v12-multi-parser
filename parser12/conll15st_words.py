#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
CoNLL15st corpus reader of words/tokens from 'pdtb-parses.json' and 'raw/'.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import json
import re


class CoNLL15stCorpus(object):
    """CoNLL15st corpus reader of words/token at document, paragraph, sentence, or word level."""

    def __init__(self, dataset_dirs, parses_ffmt=None, raw_ffmt=None, with_document=False, with_paragraph=False, with_sentence=False, paragraph_sep="^\W*\n\n\W*$", word_split=None, word_meta=False):
        if isinstance(dataset_dirs, str):
            self.dataset_dirs = [dataset_dirs]
        else:
            self.dataset_dirs = dataset_dirs
        if parses_ffmt is None:
            parses_ffmt = "{}/pdtb-parses.json"
        self.parses_ffmt = parses_ffmt
        if raw_ffmt is None:
            raw_ffmt = "{}/raw/{}"
        self.raw_ffmt = raw_ffmt
        self.with_document = with_document  # include document level list
        self.with_paragraph = with_paragraph  # include paragraph level list
        self.with_sentence = with_sentence  # include sentence level list
        self.paragraph_sep = paragraph_sep  # regex to match paragraph separator
        self.word_split = word_split  # regex to split words
        self.word_meta = word_meta  # include word metadata

    def __iter__(self):

        for dataset_dir in self.dataset_dirs:
            # next corpus
            f = open(self.parses_ffmt.format(dataset_dir), 'r')

            for line in f:
                parses_dict = json.loads(line)

                for doc_id in parses_dict:
                    # next document
                    sentences = parses_dict[doc_id]['sentences']
                    fraw = open(self.raw_ffmt.format(dataset_dir, doc_id), 'r')

                    for ret in self.process_document(doc_id, sentences, fraw):
                        yield ret

                    fraw.close()

            f.close()

    def _is_next_paragraph(self, fraw, prev_token_end, cur_token_begin):
        """Helper for paragraph boundaries using 'raw/' texts."""

        fraw.seek(prev_token_end)
        sep_str = fraw.read(cur_token_begin - prev_token_end)
        return re.match(self.paragraph_sep, sep_str, flags=re.MULTILINE)

    def _split_token(self, token):
        """Helper to split given token."""

        if re.sub(self.word_split, "", token) == "":
            return [token]
        else:
            return filter(bool, re.split(self.word_split, token))

    def process_document(self, doc_id, sentences, fraw=None):
        """Process next document."""

        document_level = []
        paragraph_id = 0  # paragraph number within document
        paragraph_level = []
        sentence_id = 0  # sentence number within document
        sentence_level = []
        token_id = 0  # token number within document

        prev_token_end = 0  # previous token last character offset

        for sentence_dict in sentences:
            sentence_token_id = token_id  # first token number in sentence

            for token in sentence_dict['words']:
                if fraw and self._is_next_paragraph(fraw, prev_token_end, token[1]['CharacterOffsetBegin']):
                    if paragraph_level:
                        if self.with_document:
                            document_level.append(paragraph_level)
                        else:
                            # yield at paragraph level
                            yield paragraph_level
                    paragraph_id += 1
                    paragraph_level = []
                prev_token_end = token[1]['CharacterOffsetEnd']

                for word in self._split_token(token[0]):
                    if not self.word_meta:
                        # return just words
                        word_level = word
                    else:
                        # return words with metadata
                        word_level = {
                            'Text': word,
                            'DocID': doc_id,
                            'ParagraphID': paragraph_id,
                            'SentenceID': sentence_id,
                            'SentenceToken': sentence_token_id,
                            'TokenList': [token_id],
                            'PartOfSpeech': token[1]['PartOfSpeech'],
                            'Linkers': token[1]['Linkers'],
                        }

                    if self.with_sentence:
                        sentence_level.append(word_level)
                    elif self.with_paragraph:
                        paragraph_level.append(word_level)
                    elif self.with_document:
                        document_level.append(word_level)
                    else:
                        # yield at word level
                        yield word_level
                token_id += 1

            if sentence_level:
                if self.with_paragraph:
                    paragraph_level.append(sentence_level)
                elif self.with_document:
                    document_level.append(sentence_level)
                else:
                    # yield at sentence level
                    yield sentence_level
            sentence_id += 1
            sentence_level = []

        if paragraph_level:
            if self.with_document:
                document_level.append(paragraph_level)
            else:
                # yield last paragraph
                yield paragraph_level

        if document_level:
            # yield at document level
            yield document_level


def load_words(dataset_dir, parses_ffmt=None, raw_ffmt=None):
    """Load all CoNLL15st corpus words by document id.

    Example output:

        all_words[doc_id][0] = {
            'Text': "Kemper",
            'DocID': doc_id,
            'ParagraphID': 0,
            'SentenceID': 0,
            'SentenceToken': 0,
            'TokenList': [0],
            'PartOfSpeech': "NNP",
            'Linkers': ["arg1_14890"],
        }
    """

    all_words_it = CoNLL15stCorpus(dataset_dir, parses_ffmt=parses_ffmt, raw_ffmt=raw_ffmt, with_document=True, with_paragraph=False, with_sentence=False, word_split="-|\\\\/", word_meta=True)

    all_words = {}
    for doc in all_words_it:
        doc_id = doc[0]['DocID']

        # store by document id
        all_words[doc_id] = doc

    return all_words


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    argp.add_argument('dataset_dir',
        help="CoNLL15st dataset directory")
    args = argp.parse_args()

    # iterate through CoNLL15st corpus at paragraph level
    it = CoNLL15stCorpus(args.dataset_dir, with_document=False, with_paragraph=True, with_sentence=False, word_split="-|\\\\/", word_meta=False)
    for item in it:
        print(item)
