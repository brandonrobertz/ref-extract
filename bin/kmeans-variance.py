#!/usr/bin/env python
from __future__ import print_function

import math
import sys
import argparse
import multiprocessing

from kmeans import cluster, extract_clusters, Xy, squared_error, cosine_error



def wssse(clusters_data, centroids, error=None):
    if not error:
        raise NotImplementedError('No error function supplied')
    wssse = 0
    for key in clusters_data.keys():
        cluster = clusters_data[key]['data']
        centroid = centroids[key]
        for point in cluster:
            # matrix is 0th item w/ extra_data 1st
            wssse += error(point, centroid)
    return wssse


def parse_args():
    desc = 'Test a series of K and compute WSSE for each, outputting in a ' \
        'gnuplot-able manner for finding optimal number of clusters.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('vec_file', type=str,
                        help='Location of word embeddings file (text vectors)')
    parser.add_argument('--max_k', type=int, default=200,
                        help='Maximum number of K to test')
    parser.add_argument('--min_k', type=int, default=1,
                        help='Minimum number of K to test')
    parser.add_argument('--step', type=int, default=1,
                        help='Number of K to step each iteration.')
    parser.add_argument('--algorithm', type=str, default='kmewans',
                        help='K-Means variant. Either kmeans or spherical. ' \
                        'Spherical K-Means uses cosine similarity to compute ' \
                        'clusters and does better in some cases.')
    args = parser.parse_args()
    return args


def test_K((k, corpus_filename, algorithm)):
    try:
        X, labels = Xy(corpus_filename)
        pred_clusters, centers = cluster(
            X,
            seed=1,
            n_clusters=k,
            alg=algorithm
        )
        clusters_data = extract_clusters(X, pred_clusters, labels, k)
        metric = wssse(clusters_data, centers)
    except Exception as e:
        metric = e
    return k, metric


if __name__ == '__main__':
    args = parse_args()
    corpus_filename = args.vec_file
    max_k = args.max_k
    step = args.step
    algorithm = args.algorithm

    pool = multiprocessing.Pool(processes=4)
    result = pool.map_async(
        test_K,
        ((k, corpus_filename, algorithm) for k in range(1, max_k, step))
    )
    for result in result.get():
        k, metric = result
        print(k, metric)
