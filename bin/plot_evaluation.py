#!/usr/bin/env python
from __future__ import print_function
import json
import argparse
import os
import re
import sys

import numpy as np


def log(msg):
    sys.stderr.write("%s\n" % msg)


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


def load_object(filename, postprocess=json.loads):
    try:
        with open(filename, 'r') as f:
            return postprocess(f.read())
    except IOError:
        print("File doesn't exist %s" % filename)
        return None


if __name__ == "__main__":
    try:
        basedir = sys.argv[1]
        outputdir = sys.argv[2]
    except IndexError:
        print("USAGE: plot_evaluation.py BASE_DIRECTORY OUTPUT_DIR")
        sys.exit(1)

    pattern = "*.json"
    datas = {}
    for fname in os.listdir(basedir):
        matches = re.match(r"(\w+)_(\w+)_(\w+)_(\w+)_(\w+)_(\w+).json", fname)
        if not matches:
            continue
        # log("File: %s" % fname)
        corpus, embedtype, kmeanstype, k, clust, _ = matches.groups()
        if int(k) > 1000:
            continue
        fname = "%s/%s" % (basedir, fname)
        log('File %s' % fname)
        try:
            data = load_object(fname)
        except ValueError as e:
            log("Error on %s: %s" % (fname, e))
            continue
        key = "%s %s" % (embedtype, kmeanstype)
        if key not in datas:
            datas[key] = {}
        means = [x["mean"] for x in data.values()]
        medians = [x["median"] for x in data.values()]
        mean = np.nanmean(means)
        median = np.nanmedian(medians)
        log('mean %s median %s' % (mean, median))
        # log('means %s medians %s' % (means, medians))
        if mean is np.nan:
            log("NAN! %s" % means)
        if median is np.nan:
            log("NAN! %s" % medians)
        datas[key][int(k)] = (mean, median)

    for key in datas:
        ks = sorted(datas[key].keys())
        name = "%s.dat" % key.replace(' ', '_')
        fileout = "%s/%s" % (outputdir, name)
        with open(fileout, "w") as f:
            f.write("# K MEAN MEDIAN\n")
            for k in ks:
                mean = datas[key][k][0]
                median = datas[key][k][1]
                f.write("%s %s %s\n" % (k, mean, median))

