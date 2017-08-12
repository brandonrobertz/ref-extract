#!/usr/bin/env python
from __future__ import print_function
import sys
import re
import argparse
import json
import multiprocessing

import numpy as np
from collections import Counter
from itertools import combinations


WINDOW_SIZE = 10
MULTIPROCESSING = True


def parse_args():
    desc = 'Compute external dataset uni and bigrams for use in PMI/coherence.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('unigram_file', type=str, help='Unigram output file')
    parser.add_argument('bigram_file', type=str, help='Bigram output file')
    args = parser.parse_args()
    return args


def update_add(dest, src):
    for k in src.keys():
        v = src[k]
        if k in dest:
            dest[k] += v
        else:
            dest[k] = v


def bigram_freq(words_window):
    """
    Get counts of sets of two words occurring in an
    array of text via a permutation (sorted by alpha).
    """
    bigrams = {}
    perms = combinations(words_window, 2)
    for w1, w2 in perms:
        if w1 == w2:
            continue
        bigram = ' '.join(sorted([w1, w2]))
        if bigram not in bigrams:
            bigrams[bigram] = 0
        bigrams[bigram] += 1
    return bigrams


def log(msg, end='\n'):
    sys.stderr.write('%s%s' % (msg, end))


def write_object(filename, data):
    with open(filename, 'w') as f:
        f.write(json.dumps(data))


def seq_process():
    unigram_freqs = {}
    bigram_freqs = {}
    N = 0
    for line in sys.stdin.readlines():
        line = line.strip()
        if not line:
            continue

        N += 1

        words = filter(lambda x: x, re.split('\s+', line))
        index = 0
        while True:
            window = words[index:index+WINDOW_SIZE]
            if len(window) < 10 and index != 0:
                break

            index += 1

            # get unigram counts
            u_freqs = Counter(window)
            update_add(unigram_freqs, u_freqs)

            # get bigram counts
            b_freqs = bigram_freq(window)
            update_add(bigram_freqs, b_freqs)

        unigrams = len(unigram_freqs.keys())
        bigrams = len(bigram_freqs.keys())
        sp = ' ' * 20
        if (N % 100) == 0:
            log("Calculating...", end='\r')
            log("# %i uni: %i bi: %s %s" % (N, unigrams, bigrams, sp), end='\r')

    return unigram_freqs, bigram_freqs


def windows():
    """
    Generate document-text windows from stdin
    """
    N = 0
    for line in sys.stdin.readlines():
        line = line.strip()
        if not line:
            continue
        N += 1
        words = filter(lambda x: x, re.split('\s+', line))
        index = 0
        while True:
            window = words[index:index+WINDOW_SIZE]
            if len(window) < 10 and index != 0:
                break
            index += 1
            sp = ' ' * 10
            if (N % 10) == 0:
                log("# %i%s" % (N, sp), end='\r')
            yield window


def process_window(words):
    # get unigram counts
    u_freqs = Counter(words)
    # get bigram counts
    b_freqs = bigram_freq(words)
    return u_freqs, b_freqs


if __name__ == "__main__":
    args = parse_args()

    # pool = multiprocessing.Pool()
    # results = pool.map(process_window, windows())
    # pool.close()
    # pool.join()
    # print("Done!")
    # print("Results", results)

    # non-parallel
    unigram_freqs, bigram_freqs = seq_process()

    log('Done!')
    write_object(args.unigram_file, unigram_freqs)
    write_object(args.bigram_file, bigram_freqs)

