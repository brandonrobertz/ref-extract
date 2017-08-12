#!/usr/bin/env python
from __future__ import print_function
import json
import copy
import sys
import argparse
import re

import numpy as np


basedir = "tmp_data/evaluation"


def parse_args():
    desc = 'Find highest and worse performing clusters'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'k', type=str, default='600',
        help='Number of K to check or a bash-style range (start..end[..step])' \
        ' e.g., 100..1000..10, meaning 100 through 1000 k, with a step of 10.' \
        ' Note that step is optional and will default to 1.'
    )
    parser.add_argument(
        '--output', type=str, default='all',
        help='Output type: all or combined'
    )
    args = parser.parse_args()
    return args


def parse_range(range_str):
    if '..' not in range_str:
        return [int(range_str)]
    match_step = re.match('([0-9]+)\.\.([0-9]+)\.\.([0-9]+)', range_str)
    match_range = re.match('([0-9]+)\.\.([0-9]+)', range_str)

    if not match_range and not match_step:
        raise ValueError('Invalid range supplied %s' % range_str)

    start_k = None
    end_k = None
    step = 1
    if match_step:
        start_k, end_k, step = match_step.groups()
    elif match_range:
        start_k, end_k = match_range.groups()

    return range(int(start_k), int(end_k), int(step))


def log(msg):
    sys.stderr.write("%s\n" % msg)


def load_object(filename, postprocess=json.loads):
    log("Loading object %s" % filename)
    with open(filename, 'r') as f:
        return postprocess(f.read())


def process_csv(csv_string):
    csv_lines = csv_string.strip().split('\n')
    lines = filter( lambda x: x, csv_string.split('\n'))
    data = {}
    for line in lines:
        cluster, words = line.split(',')
        data[cluster.strip()] = filter(lambda x: x, words.strip().split())
    return data


if __name__ == "__main__":
    args = parse_args()
    k_range = parse_range(args.k)

    models = [
        "fasttext_spherical",
        "word2vec_spherical",
        "fasttext_kmeans",
        "word2vec_kmeans",
        "lda_lda"
    ]
    template = {
        "eval": "%(base)s/fulltext_%(model)s_%(k)s_clusters_evaluation.json",
        "clust": "tmp_data/fulltext_%(model)s_%(k)s_clusters/clusters.csv"
    }

    cluster_scores = []
    for k in k_range:
        for model in models:
            filename_args = {
                "k": k,
                "base": basedir,
                "model": model
            }
            evalfile = template["eval"] % (filename_args)
            clustfile = template["clust"] % (filename_args)
            evals = load_object(evalfile)
            clusters = load_object(clustfile, postprocess=process_csv)
            for cluster in evals.keys():
                score = evals[cluster]['mean']
                cluster_labels = ", ".join(clusters[cluster])
                # print('cluster_eval %s' % cluster_eval)
                # print('cluster_labels %s' % cluster_labels)
                score_name = '%s-%s' % (model, k)
                if np.isnan(score):
                    continue
                cluster_scores.append([score, score_name, cluster_labels])

    seen_identifiers = {}
    for score, cluster_name, labels in sorted(cluster_scores):
        if args.output == 'all':
            print('%s %s %s' % (score, cluster_name, labels))
            continue

        unique_identifier = '%s %s' % (score, labels)
        if unique_identifier not in seen_identifiers:
            seen_identifiers[unique_identifier] = []
        seen_identifiers[unique_identifier].append(cluster_name)

    if args.output == 'combined':
        for identifier in sorted(seen_identifiers.keys()):
            score, labels = identifier.split(' ', 1)
            cluster_names = seen_identifiers[identifier]
            print('%s (%s) [%s]' % (score, labels, ' '.join(cluster_names)))

