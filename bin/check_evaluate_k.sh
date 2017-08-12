#!/bin/bash

# generate all the clusters
for VARIANT in spherical kmeans; do
  for ALG in word2vec fasttext; do
    # build clusters
    for K in {100..2000..10}; do
      F="tmp_data/fulltext_${ALG}_${VARIANT}_${K}_clusters/clusters.csv"
      if [ ! -e $F ] || [ `wc -l $F | awk '{print $1}'` -lt 10 ]; then
        echo $VARIANT cluster $K for $ALG doesnt exist
        wc -l $F 2> /dev/null
      fi
    done
  done
done
