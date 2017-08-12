#!/usr/bin/env python2
from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from spherecluster import SphericalKMeans
from matplotlib import offsetbox

from jinja2 import Template

import matplotlib.pyplot as plt
import numpy as np
import fileinput
import sys
import json
import argparse
import math


def squared_error(point, centroid):
    return np.sqrt(np.sum(np.power(point - centroid, 2)))


def cosine_error(point, centroid):
    p = np.sqrt(np.sum(np.power(np.abs(point), 2)))
    c = np.sqrt(np.sum(np.power(np.abs(centroid), 2)))
    pc = np.sum(np.abs(point.dot(centroid.transpose())))
    return  pc / ( p * c)


def const_from_file(filename, default=None):
    try:
        with open(filename) as f:
            return int(f.read())
    except Exception, e:
        print( "Error reading constant file {}: {}".format(
            filename, e
        ))
    return default


def log(msg):
    sys.stderr.write(msg)
    sys.stderr.write('\n')


def vectorize_line(line):
    """ Take a line of fastText text-vectors file (.vec) and output a
    label(word) and the vector as (X, label)
    """
    vectors = line.split()
    label = vectors.pop(0)
    vectors = map( lambda x: float(x), vectors)
    # go through stdin reading fastText vectors and add them to array
    X = np.array(vectors)
    return X, label


def closest_label(X, labels, vec, dist='cosine', ooc_only=False, top=10):
    if dist == 'euclidean':
        sim = euclidean_distances(X, vec.reshape(1, -1))
    elif dist == 'cosine':
        sim = cosine_similarity(X, vec.reshape(1, -1))
    else:
        raise NotImplementedError('dist must be euclidean or cosine')
    # get the top five indices
    indices = sim.argsort(axis=0)[-top:][::-1]
    words = []
    for i in indices:
        words.append(labels[i[0]])
    return " ".join(words)


def Xy(filename):
    """ Read from filename, grabbing all vectors from our incoming stream of
    fastTex vector (text) file. Return matrices (X, labels)
    """
    fi = fileinput.input(files=filename)
    header = fi.readline()
    X = []
    labels = []
    for line in fi:
        x, label = vectorize_line(line)
        X.append(x)
        labels.append(label)
    return np.array(X), labels


def cluster(X, seed=0, n_clusters=20, alg='kmeans'):
    """
    Perform k-means on given X data. For alg, use one of:
    'kmeans' (sklearn KMeans) or 'spherical' (SphericalKMeans)
    returns (X pred clusters, cluster centers)
    NOTE: euclidean tends to perform very poorly
    """
    # log("Clustering k-means with {} clusters".format(n_clusters))
    if alg == 'kmeans':
        Model = KMeans
    elif alg == 'spherical':
        # inplace l2 normalization (spherical k-means assumes this)
        normalize(X, 'l2', copy=False)
        Model = SphericalKMeans

    kmeans = Model(
        n_clusters=int(n_clusters), random_state=seed
    )
    pred_clusters = kmeans.fit_predict(X)
    return pred_clusters, kmeans.cluster_centers_


def plot_cluster(cluster, X, labels, seed=0, outdir=None):
    # TODO: plot spherical using the spherical plot

    model = TSNE(n_components=2, init='pca', random_state=seed)

    log("%s Cluster %s %s" % ('-' * 20, cluster, '-' * 20))
    log("Fitting t-SNE model on cluster")
    tsne_X = model.fit_transform(X)

    x_min, x_max = np.min(tsne_X, 0), np.max(tsne_X, 0)

    # normalize
    if ((x_max - x_min) != np.array([0., 0.])).all():
        tsne_X = (tsne_X - x_min) / (x_max - x_min)

    mult = 2
    plt.figure(figsize=(8 * mult, 6 * mult), frameon=False)
    ax = plt.subplot()

    Xy = [[100, 7], [33, 8], [5, 10]]
    X = map(lambda x: [x[0]], Xy)
    y = map(lambda x: x[1], Xy)
    model = LinearRegression().fit(X, y)
    fontsize = np.round(model.intercept_ + model.coef_[0] * tsne_X.shape[0])
    fontsize = fontsize if fontsize > 6 else 6
    fontsize = fontsize if fontsize < 13 else 13
    print( "Using font", fontsize, "for", tsne_X.shape[0], "words")

    log("Plotting embeddings")
    for i in range(tsne_X.shape[0]):
        plt.text(
            tsne_X[i, 0], tsne_X[i, 1],
            str(labels[i]),
            #color=plt.cm.gist_earth(cluster / TOTAL_CLUSTERS),
            fontdict={'weight': 'regular', 'size': fontsize})

	ax.axis('off')
    plt.title("Cluster {} - {} Items".format(cluster, tsne_X.shape[0]))

    log("Building figure")
    plt.savefig('{}/{}.svg'.format(outdir, cluster), format='svg', dpi=1200)
    if cluster == 0:
		plt.show()

    plt.close()


def plot_embedding(clusters, title=None, seed=0, n_clusters=20, outdir=None):
    """ Scale and visualize the embedding vectors
    """
    log("Plotting each cluster")
    for c in clusters.keys():
        plot_cluster(
            c,
            clusters[c]['data'],
            clusters[c]['labels'],
            seed=seed,
            outdir=outdir
        )


def extract_clusters(X, pred_clusters, labels, n_clusters):
    """ Take our vectors, predicted cluster memberships, labels (words)
    and (as an optimization) the number of clusters, and create a dict
    of each cluster containing the vectors and labels.
    Returns: dict in the following format, where keys are cluster IDs
    { 0: { 'data': [x1, ..., xN], 'labels': [label1, ..., labelN] }, ...}
    """
    clusters = {
        x: {'data': [], 'labels': []}
        for x in range(int(n_clusters))
    }
    # log("Splitting data by cluster")
    for i in range(len(X)):
        data = X[i]
        pred_cluster = pred_clusters[i]
        label = labels[i]

        clusters[pred_cluster]['data'].append(data)
        clusters[pred_cluster]['labels'].append(label)

    return clusters


def error(point, centroid, losstype='euclidean'):
    error = None
    if losstype == 'euclidean':
        error = squared_error
    math.sqrt(
        sum([x**2 for x in (point - centroid)]))


def wssse(clusters_data, centroids):
    avgs = [0] * len(clusters_data.keys())
    errs = [0] * len(clusters_data.keys())
    totals = [0] * len(clusters_data.keys())
    for key in clusters_data.keys():
        cluster = clusters_data[key]['data']
        centroid = centroids[key]
        for point in cluster:
            # matrix is 0th item w/ extra_data 1st
            errs[key] += squared_error(point, centroid)
            totals[key] += 1
    for key in clusters_data.keys():
        avgs[key] = errs[key] / float(totals[key])

    return avgs


def write_cluster_errs(centroids, clusters_data, outdir=None):
    with open('%s/cluster_errs.csv' % outdir, 'w') as f:
        wssses = wssse(clusters_data, centroids)
        for i in range(len(centroids)):
            center = centers[i]
            err = wssses[i]
            labels = clusters_data[i]['labels']
            f.write("%s, %s, %s\n" % ( i, err, " ".join(labels)))


def write_clusters(clusters, outdir=None):
    with open('%s/clusters.csv' % outdir, 'w') as f:
        for c in clusters.keys():
            cluster = clusters[c]
            data = cluster['data']
            labels = cluster['labels']
            f.write("%s, %s\n" % (c, ' '.join(labels)))


def write_centroids(centroids, u_X, u_labels, outdir=None, output=True):
    with open('%s/centroid_labels.csv' % outdir, 'w') as f:
        for i in range(len(centroids)):
            center = centers[i]
            labels = closest_label(u_X, u_labels, center, dist=args.distance)
            if output:
                print(i, labels)
            f.write("%s, %s\n" % ( i, labels))


def render_index(n_clusters=20, outdir=None):
    with open('%s/index.tpl.html' % outdir) as f:
        index_template = f.read()
    tpl = Template(index_template)
    rendered = tpl.render(clusters=range(n_clusters))
    with open('%s/index.html' % outdir, 'w') as f:
        f.write(rendered)


def parse_args():
    desc = 'Cluster word embeddings with k-means. Output centroid word ' \
        'similarities, cluster assignments, and build cluster visualizations.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('infile', type=str,
                        help='Location of word embeddings file (text vectors)')
    parser.add_argument('uinfile', type=str,
                        help='Location of unique word embeddings file.')
    parser.add_argument('clusters', type=int,
                        help='Number of clusters.')
    parser.add_argument('outdir', type=str,
                        help='Directory to write clusters and related data. ' \
                        'This directory must exist.')
    parser.add_argument('--algorithm', type=str, default='kmewans',
                        help='K-Means variant. Either kmeans or spherical. ' \
                        'Spherical K-Means uses cosine similarity to compute ' \
                        'clusters and does better in some cases.')
    parser.add_argument('--distance', type=str, default='cosine',
                        help='Distance measure for extrapolating closest ' \
                        'words to cluster centroids. Either cosine or ' \
                        'euclidean. NOTE: euclidean distance performs ' \
                        'poorly in most cases.')
    parser.add_argument('--output', type=bool, default=False,
                        help='Whether to print out the cluster summaries.')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Whether to plot the clusters using t-SNE.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    u_X, u_labels = Xy(args.uinfile)
    X, labels = Xy(args.infile)
    pred_clusters, centers = cluster(
        X,
        seed=7877,
        n_clusters=args.clusters,
        alg=args.algorithm
    )
    clusters_data = extract_clusters(X, pred_clusters, labels, args.clusters)

    if args.plot:
        try:
            plot_embedding(
                clusters_data,
                "t-SNE embedding of the clusters",
                n_clusters=args.clusters,
                outdir=args.outdir
            )
        except Exception, e:
            print('Error plotting embedding clusters', e)

    # render_index(n_clusters=N_CLUSTERS, outdir=args.outdir)
    write_clusters(clusters_data, outdir=args.outdir)
    write_centroids(
        centers, u_X, u_labels, outdir=args.outdir, output=args.output
    )
    write_cluster_errs(centers, clusters_data, outdir=args.outdir)

    for i in range(len(centers)):
        c = clusters_data[i]
        labels = c["labels"]
        center = centers[i]
        closest_labels = closest_label(
            u_X, u_labels, center, dist=args.distance, ooc_only=True
        )
        print('-' * 20, "cluster", i, '-' * 20)
        print('Labels:\n', " ".join( labels))
        print('Nearest:\n', closest_labels)
