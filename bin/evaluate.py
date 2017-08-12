#!/usr/bin/env python
from __future__ import print_function
import sys
import json
import argparse

from itertools import combinations
import numpy as np


def parse_args():
    desc = 'Evaluate a topic model using windowed external PMI'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('unigram_file', type=str,
                        help='Location of unigram_freqs.json')
    parser.add_argument('bigram_file', type=str,
                        help='Location of bigram_freqs.json')
    parser.add_argument('topic_file', type=str,
                        help='Location of topic words CSV ' \
                        '(centroid_labels.csv)')
    args = parser.parse_args()
    return args


def log(msg, end='\n'):
    sys.stderr.write('%s%s' % (msg, end))


def load_object(filename, postprocess=json.loads):
    try:
        with open(filename, 'r') as f:
            return postprocess(f.read())
    except IOError:
        print("File doesn't exist %s" % filename)
        return None


def process_csv(csv_string):
    csv_lines = csv_string.strip().split('\n')
    lines = filter( lambda x: x, csv_string.split('\n'))
    data = {}
    for line in lines:
        cluster, words = line.split(',')
        data[cluster.strip()] = filter(lambda x: x, words.strip().split())
    return data


# PMI = median( log(p(a,b) / ( p(a) * p(b) ))) for all combinations in topic
def pmi(word1, word2, unigram_freqs, unigram_sum, bigram_freqs, bigram_sum):
    try:
        p_word1 = unigram_freqs[word1] / unigram_sum
    except Exception as e:
        log('word not found %s' % word1)
        p_word1 = 0.0
    try:
        p_word2 = unigram_freqs[word2] / unigram_sum
    except Exception as e:
        log('word not found %s' % word2)
        p_word2 = 0.0
    bigram = ' '.join(sorted([word1, word2]))
    try:
        p_bigram = bigram_freqs[bigram] / bigram_sum
    except Exception as e:
        # log('bigram not found %s' % bigram)
        p_bigram = 0.0
    p_w1_w2 = ( p_word1 * p_word2)
    if not p_w1_w2:
        return 0
    val = p_bigram / p_w1_w2
    if not val:
        return 0
    return np.log( val)


if __name__ == "__main__":
    args = parse_args()

    # load unigram
    unigram_freqs = load_object(args.unigram_file)
    # log('loaded unigram freqs from %s' % args.unigram_file)

    # load bigram
    bigram_freqs = load_object(args.bigram_file)
    # log('loaded bigram freqs from %s' % args.bigram_file)

    # load topic model output
    topics = load_object(args.topic_file, postprocess=process_csv)
    # log('loaded topics from %s' % args.topic_file)
    if not topics:
        sys.exit(0)

    # only do this sum once
    unigram_sum = float(np.sum(unigram_freqs.values()))
    bigram_sum = float(np.sum(bigram_freqs.values()))
    # log('unigrams: %i bigrams: %i' % (unigram_sum, bigram_sum))

    topic_pmis = { k: None for k in topics.keys()}
    for topic in topics.keys():
        words = topics[topic]
        # 1. get every combination
        pairs = combinations(words, 2)
        # 2. compute PMI for each
        pmis = []
        for word1, word2 in pairs:
            value = pmi(
                word1, word2,
                unigram_freqs, unigram_sum,
                bigram_freqs, bigram_sum
            )
            pmis.append(value)
        # 3. take median of all PMIs for topic
        # med = np.median(pmis)
        # log('topic: %s words: %s pmis: %s' % (topic, ','.join(words), pmis))
        # log('topic: %s words: %s' % (topic, ','.join(words)))
        mean = np.nanmean(pmis)
        median = np.nanmedian(pmis)
        topic_pmis[topic] = {'mean':mean, 'median':median}

    topic_keys = sorted(topic_pmis.keys())
    # log('topic_keys', topic_keys)
    # log('topic_pmis %s' % topic_pmis)
    # save the topic pmis
    print( json.dumps(topic_pmis))

