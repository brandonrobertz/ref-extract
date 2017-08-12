#!/usr/bin/env python2
from __future__ import print_function
import sys
import re
import json
import time
import argparse
import ciso8601
import pandas as pd
import multiprocessing

from itertools import combinations


def log(msg, err=True, end='\n'):
    if err:
        sys.stderr.write(msg)
        sys.stderr.write(end)


def read_stopwords(filename):
    with open(filename) as f:
        words = f.readlines()
    return map(lambda w: w.lower().strip, words)


def read_clusters_txt(filename):
    with open(filename) as f:
        lines = f.readlines()

    clusters = []
    for line in lines:
        if not line:
            continue
        clusters.append([ w.strip().replace(',', '') for w in line.split()])

    return clusters


def date_str_to_timestamp(date):
    t = ciso8601.parse_datetime(date)
    if not t:
        log("FAILED PARSING TIME {}".format(date))
        return None
    return time.mktime(t.timetuple())


def date_from_title(title):
    """ The SRT filenames are in the format of MM-DD-YYYY mostly with some variation.
    """
    # these are also in month, day, year format (US format)
    replacements = {
        "tmp_data/transcripts/Donald Trump's Full Indiana Victory Speech-5oO4YqOGClY.en.vtt.srt":
        ('5', '3', '2015'),
        "tmp_data/transcripts/Donald Trump's Full Indiana Primary Victory Speech-SORp0GmgLX0.en.vtt.srt":
        ('5', '4', '2015'),
        "tmp_data/transcripts/Full Espeech - Donald Trump Rally in Wilmington, Ohio (September 1, 2016) Trump LiveSpeech-tUjsvBHmPj4.en.vtt.srt":
        ('9', '1', '2016'),
        "tmp_data/transcripts/Full Speech - Donald Trump Rally in Akron, Ohio (August 22, 2016) Donald Trump Ohio Speech-9BvdJ3Q7GzQ.en.vtt.srt":
        ('8', '22', '2016'),
        "tmp_data/transcripts/FULL Event HD - Donald Trump Rally in Knoxville, TN w_ Beada Corum Intro-gJuWzsnh_Xw.en.vtt.srt":
        ('11', '16', '2015'),
        "tmp_data/transcripts/Full Speech - Donald Trump AMAZING Speech at Charter School in Cleveland, OH (September 8, 2016)-TR_1F3xd-Ng.en.vtt.srt":
        ('8', '2', '2016'),
        "tmp_data/transcripts/Donald Trump - THE WHOLE SYSTEM IS RIGGED; RNC Should be Ashamed-UqByMThBpfs.en.vtt.srt":
        ('4', '12', '2016'),
        "tmp_data/transcripts/Full Event - Donald Trump Town Hall in Hickory, NC at Lenoir-Rhyne University (3-14-1-_cOlpRxtIB4.en.vtt.srt":
        ('14', '3', '16'),
        "tmp_data/transcripts/FULL Speech HD - Donald Trump Speaks to 10,000 Plus in Lowell, MA (1-4-16)-v6LYm17J7ec.en.vtt.srt":
        ('4', '1', '2016'),
        "tmp_data/transcripts/Full Event - Donald Trump Rally in Jackson, Mississippi (August 24, 2016) Trump Live Speech-VVBuT39Bc1I.en.vtt.srt":
        ('8', '24', '2016'),
		"tmp_data/transcripts/Full Event - Donald Trump Town Hall in Hickory, NC at Lenoir-Rhyne University (3-14-1-_cOlpRxtIB4.en.vtt.srt":
		('3', '14', '2016'),
		"tmp_data/transcripts/FULL SPEECH - Donald Trump Rally in Greenville, North Carolina (September 6, 2016) Trump Live Speech-Fj_5DIJy-zE.en.vtt.srt":
		('9', '6', '2016'),
		"tmp_data/transcripts/Full Event - Donald Trump Speech at 'Roast and Ride' Event in Des Moines, Iowa (August 27, 2016)-uqOMY4YuoKE.en.vtt.srt":
		('9', '27', '2016'),
		"tmp_data/transcripts/Full Espeech - Donald Trump Rally in Wilmington, Ohio (September 1, 2016) Trump Live Speech-tUjsvBHmPj4.en.vtt.srt":
		('9', '1', '2016'),
		"tmp_data/transcripts/FULL Speech HD - Donald Trump Speaks to 10,000 Plus in Lowell, MA (1-4-16)-v6LYm17J7ec.en.vtt.srt":
		('1', '4', '2016'),
	}

    title = title.strip()
    found_date = re.findall(r'([0-9]+).([0-9]+).([0-9]+)', title)

    if replacements.get(title):
        log("Found match for %s" % title)
        found_date = [replacements[title]]

    if not found_date or len(found_date) < 1:
        log( "CAN'T CONVERT: {}".format( title))
        return None

    date = found_date[0]

    year = date[2]
    if len(year) == 2:
        year = "20" + year

    if int(year) < 2014:
        log( "Bad year: {} for title: {}".format( date, title))
        return None

    month = date[0]
    day = date[1]
    if int(month) > 12:
        log( "Bad month: {} for title: {}".format( month, title))
        month = date[1]
        day = date[0]

    ymd = "{}-{}-{}".format(year, month.zfill(2), day.zfill(2))

    return ymd


def extract_dates(titles_file):
    """
    Returns document-indexed list of dates in the following format:

        ["YYYY-MM-DD", "YYYY-MM-DD", None, ...]

    None, indicating a document without a date.
    """
    dates = []
    with open(titles_file) as f:
        lines = f.readlines()
        for title in lines:
            dates.append(date_from_title(title))
    return dates


def read_documents(documents_file):
    """
    """
    documents = []
    with open(documents_file) as f:
        lines = f.readlines()
        log('Total lines in documents file %i' % len(lines))
        for doc in lines:
            documents.append(doc)
    return documents


def windows(document, window_size=10):
    """
    Generate document-text windows from stdin
    """
    N = 0
    words = filter(lambda x: x, re.split('\s+', document.strip()))
    index = 0
    while True:
        window = words[index:index+window_size]
        if len(window) < 10 and index != 0:
            break
        index += 1
        sp = ' ' * 10
        # if (N % 10) == 0:
        # log("# %i%s" % (N, sp), end='\r')
        yield window


def windowed_bigram_counts(clusters_words, document, doc_ix=1):
    counts = {}

    e = enumerate(document.split())
    needles = [ i for i, w in e if w == w1 or w == w2]

    for cluster in clusters_words.keys():
        words = { w.strip() for w in clusters_words[cluster]}

        if cluster not in counts:
            counts[cluster] = 0

        N = -1
        for w1, w2 in combinations(words, 2):
            N += 1

            if w1 not in document or w2 not in document:
                continue

            if N % 100000 == 0:
                log("# %i %i          " % (doc_ix, N), end='\r')

            if w1 in window_words and w2 in window_words:
                counts[cluster] += 1

    return counts


def ts_to_df(ts, cid):
    df = pd.DataFrame.from_records(ts)
    ix = pd.DatetimeIndex(df['date']).unique()
    df.set_index(ix, inplace=True)
    df.drop('date', 1, inplace=True)
    return df


def title_from_wordfile(filename, delim=' ', insert_breaks=True):
    labels = [""] * total_clusters
    lens = [0] * total_clusters
    with open(filename, 'r') as f:
        for line in f.readlines():
            c, label_str = line.split(delim, 1)
            cid = int(c)
            label_str = label_str.strip()
            if len(labels[cid]) > (120 * 3):
                label_str = '.'
            lens[cid] += len(label_str)
            labels[cid] += label_str
            if insert_breaks and lens[cid] > 120:
                labels[cid] += "\\n"
                lens[cid] = 0
            else:
                labels[cid] += " "
    return labels


def write_gnuplot_file(prefix,
                       total_clusters,
                       cluster_labels_csv,
                       clusters,
                       output_dir):
    log('Writing gnuplot file')

    cent_labels = title_from_wordfile(
        cluster_labels_csv, delim=',', insert_breaks=False
    )
    log('cent_labels', cent_labels)

    labels = title_from_wordfile(clusters)
    log('labels', labels)

    script = """set term png size 1000,{}
set output "{}/{}.png"
unset key
set grid xtics ytics ls 3 lw 1 lc rgb 'gray'
set multiplot layout {},1
set datafile separator ","
set ylabel 'Topic Words' font ', 9'
set xlabel 'Speech Date' font ', 9'
set xdata time
set format x "%m/%y"
set timefmt "%Y-%m-%d"
""".format(total_clusters * 300, output_dir, prefix, total_clusters)

    for i in range(total_clusters):
        script += 'set title "%s\\n%s" font "arial,8"\n' % (
            cent_labels[i], labels[i].strip()
        )
        script += 'plot "%s/%s.timeseries.csv" ' \
                'using 1:%i w lines\n' % (output_dir, prefix, i + 2)
    with open('%s/%s.gnuplot' % (output_dir, prefix), 'w') as f:
        f.write(script)


def parse_args():
    desc = 'Generate a precalculated list of timeseries data for given ' \
        'clusters. This step is quite time consuming so we typically ' \
        'only want to do this once and store the result. Outputs final ' \
        'JSON object to STDOUT.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('clusters_file', type=str,
                        help='Location of clusters.txt file')
    parser.add_argument('corpus_file', type=str,
                        help='Location of preprocessed corpus')
    parser.add_argument('titles_file', type=str,
                        help='Location of titles (CORPUS.titles) file')
    args = parser.parse_args()
    return args


def f(cluster):
    if len(cluster) > 15:
        return

    cluster_date_scores = []

    #   INNER LOOP:
    #     a. for each combination pair
    for w1, w2 in combinations(cluster, 2):

        #     b. for each document
        cluster_date_scores = []
        score = 0
        for i in range(len(doc_words_ix_index)):

            doc_words = doc_words_ix_index[i]["words_ix"]
            # log('doc_words: %s' % doc_words)
            doc_date = doc_words_ix_index[i]["date"]

            # if not doc_date:
            #     continue

            if (w2 not in doc_words.keys()) or (w1 not in doc_words.keys()):
                # cluster_date_scores.append({"date": doc_date, "score": score})
                continue

            #     c. get each index pairs, filter out all > abs(10) apart
            w1_indices = doc_words[w1]
            w2_indices = doc_words[w2]
            for w1i in w1_indices:
                for w2i in w2_indices:
                    if abs(w1i - w2i) <= 10:
                        score += 1
                        log('w1 %s w2 %s increasing cluster score %i' % (w1, w2, score))

            if score > 0:
                cluster_date_scores.append({"date": doc_date, "score": score})

    # print("%s|%s" % ( cluster, json.dumps(cluster_date_scores)))
    return {"cluster": cluster, "scores": cluster_date_scores}


# load clusters into memory
# go through each fulltext document doing:
#  - getting counts for each cluster
#  - get date of fulltext file
# output time series json
if __name__ == "__main__":
    args = parse_args()
    CLUSTERS_TXT=args.clusters_file
    FULLTEXT_FILE=args.corpus_file
    TITLES_FILE=args.titles_file

    # 1. remove/ignore all documents without dates to reduce workload
    dates = extract_dates(TITLES_FILE)
    documents = read_documents(FULLTEXT_FILE)

    log('N dates %i N docs %i' % (len(dates), len(documents)))

    # 2. build index of document -> words -> indices
    log('Building document -> words -> indices index')
    doc_words_ix_index = []
    for i in range(len(dates)):
        document = documents[i]
        date = dates[i]
        words = [ w.strip() for w in document.split() ]
        words_ix = {}
        for ix, word in enumerate(words):
            if word not in words_ix.keys():
                words_ix[word] = []
            words_ix[word].append(ix)

        doc_words_ix_index.append({
            "date": date,
            "words_ix": words_ix
        })

    # 3. for each cluster, get combinations
    log('Reading clusters file %s' % CLUSTERS_TXT)
    clusters = read_clusters_txt(CLUSTERS_TXT)
    clusters_date_scores = []

    pool = multiprocessing.Pool()
    clusters_date_scores = pool.map(f, clusters)

    #     d. count remaining and build this --
    #     e. cluster -> dates -> counts

    # print("cluster_words %s" % cluster_words)
    # print("dates %s" % dates)
    # print("dates_index %s" % dates_index)
    log('Done!')
    print("clusters_date_scores\n%s" % json.dumps(clusters_date_scores))

