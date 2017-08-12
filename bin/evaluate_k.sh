#!/bin/bash

CMD=${1}

if [[ "${CMD}" == "" ]]; then
    echo 'USAGE: evaluate_k.sh [cluster|eval]'
fi

# generate all the clusters
for VARIANT in spherical kmeans; do
    for ALG in word2vec fasttext; do
        if ! [ -f tmp_data/fulltext.$ALG.keywords.vec ]; then
            # build our embeddings
            make keywords.uniq.vec EMBED_PROGRAM="$ALG" K_MEANS_VARIANT="$VARIANT"
        fi

        # build clusters
        if [[ "${CMD}" == "cluster" ]]; then
            for K in {100..1000..5}; do
                if [ -e tmp_data/fulltext_${ALG}_${VARIANT}_${K}_clusters/clusters.csv ] && [ `wc -l tmp_data/fulltext_${ALG}_${VARIANT}_${K}_clusters/clusters.csv | awk '{print $1}'` -gt 10 ]; then
                    echo "$ALG cluster $K exist! Skipping."
                    continue
                fi
                make clusters N_CLUSTERS="$K" EMBED_PROGRAM="$ALG" K_MEANS_VARIANT="$VARIANT"
            done
        fi

        # get evaulation data
        if [[ "${CMD}" = "eval" ]]; then
            ## word vectors
            for K in {100..1000..5}; do
                if [ ! -e tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_clusters_evaluation.json ]; then
                    echo Evaluating clusters ${K} for ${ALG} ${VARIANT}
                    ./bin/evaluate.py \
                        tmp_data/evaluation/fulltext_unigrams.json \
                        tmp_data/evaluation/fulltext_bigrams.json \
                        tmp_data/fulltext_${ALG}_${VARIANT}_${K}_clusters/clusters.csv \
                        > tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_clusters_evaluation.json
				else
					wc -l tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_clusters_evaluation.json
                fi
                if [ ! -e tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_centroids_evaluation.json ]; then
                    echo Evaluating centroids ${K} for ${ALG} ${VARIANT}
                    ./bin/evaluate.py \
                        tmp_data/evaluation/fulltext_unigrams.json \
                        tmp_data/evaluation/fulltext_bigrams.json \
                        tmp_data/fulltext_${ALG}_${VARIANT}_${K}_clusters/centroid_labels.csv \
                        > tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_centroids_evaluation.json
				else
					wc -l tmp_data/evaluation/fulltext_${ALG}_${VARIANT}_${K}_centroids_evaluation.json
                fi
            done
        fi

    done
done

# do LDA comparison eval
if [[ "${CMD}" = "eval" ]]; then
    for K in {100..1000..5}; do
        if [ ! -e tmp_data/evaluation/fulltext_lda_lda_${K}_centroids_evaluation.json ]; then
            echo Evaluating LDA ${K}
            ./bin/evaluate.py \
                tmp_data/evaluation/fulltext_unigrams.json \
                tmp_data/evaluation/fulltext_bigrams.json \
                tmp_data/lda_evaluation_centroids/${K}/fulltext.${K}.csv \
                > tmp_data/evaluation/fulltext_lda_lda_${K}_centroids_evaluation.json
        fi
        if [ ! -e tmp_data/evaluation/fulltext_lda_lda_${K}_clusters_evaluation.json ]; then
            echo Evaluating LDA ${K}
            ./bin/evaluate.py \
                tmp_data/evaluation/fulltext_unigrams.json \
                tmp_data/evaluation/fulltext_bigrams.json \
                tmp_data/lda_evaluation_clusters/${K}/fulltext.${K}.csv \
                > tmp_data/evaluation/fulltext_lda_lda_${K}_clusters_evaluation.json
        fi
    done
fi
