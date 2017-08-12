#!/usr/bin/env python2
from __future__ import print_function
import sys
import json
import argparse
import pandas as pd


def log(msg, err=True, end='\n'):
    if err:
        sys.stderr.write(msg)
        sys.stderr.write(end)


def parse_args():
    desc = 'Load our preprocessed timeseries cluster data and extract ' \
        'clusters based on query conditions. Output as gnuplot ' \
        'compatable DAT file.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('timeseries_json', type=str,
        help='Location of ts_output.json file (from timeseries_preprocess.py)'
    )
    parser.add_argument('--query', type=str,
        help='Query for clusters. If a query word is found in a cluster, it '
             'will be included in timeseries DAT output. For multiple matching '
             'words, comma separate words (this ANDs the cluster query).'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.timeseries_json, 'r') as f:
        ts = json.loads(f.readlines()[1])

    # remove null results
    ts_data = filter(lambda x: x, ts)
    log('Total records read %i' % len(ts_data))

    queries = [ q.strip() for q in args.query.split(',')]
    log('Using queries %s' % queries)

    for query in queries:
        log('Query %s' % query)
        ts_data = filter(lambda x: query in x['cluster'], ts_data)

    log('Total results %i' % len(ts_data))

    # print('date', 'citations')
    records = []
    for data in ts_data:
        log('Result %s' % data)
        # cluster = '-'.join(data['cluster'])
        # columns.append(cluster)
        scoredatas = data['scores']
        for sdata in scoredatas:
            date = sdata['date']
            if not date:
                log('No date for cluster %s!' % sdata)
                continue
            scores = sdata['score']
            print('%s %s' % (date, scores))

    #     if not date or not score['score']:
    #         continue
    #     if date not in timeseries:
    #         timeseries[date] = {}
    #     for cid in counts.keys():
    #         label = 'Cluster-%02i' % cid
    #         if label not in timeseries[date]:
    #             timeseries[date][label] = 0.0
    #         timeseries[date][label] += counts[cid]

    # timeseries_flat = []
    # for date in timeseries:
    #     records = timeseries[date]
    #     records['date'] = date
    #     timeseries_flat.append(records)

    # # we need a column for each cluster, indexed by date
    # df = pd.DataFrame.from_records(timeseries_flat)
    # df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
    # df.drop('date', 1, inplace=True)
    # df.fillna(0, inplace=True)
    # df.sort_index(inplace=True)

    # print( df.resample('W').sum().to_csv())

    # if args.output_dir:
    #     write_gnuplot_file(
    #         args.prefix, total_clusters,
    #         args.centroid_labels_csv,
    #         args.clusters_txt,
    #         args.output_dir
    #     )
