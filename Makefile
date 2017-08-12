# File must be in tmp_data/CORPUS
CORPUS=fulltext
EMBEDTYPE=skipgram
EMBED_PROGRAM=fasttext
K_MEANS_VARIANT=spherical
DISTANCE=cosine
N_CLUSTERS=
PCT_KEYWORDS=0.10

# housekeeping
CLUSTERS_DIR=${CORPUS}_${EMBED_PROGRAM}_${K_MEANS_VARIANT}_${N_CLUSTERS}_clusters

# Fulltext political transcripts parameters
# make clusters CORPUS=fulltext PCT_KEYWORDS=0.08 EMBEDTYPE=cbow N_CLUSTERS=33

################################################################################
# GOP debate data
################################################################################
gop_debates:
	rm -rf tmp_data/gop_debates
	mkdir -p tmp_data/gop_debates/raw
	./bin/scrape_gop_debates.sh tmp_data/gop_debates/raw data/gop_debates/list.txt
	find tmp_data/gop_debates/ -type f -exec cat {} \; \
			| awk '{print $$1}' \
			| sort \
			| uniq \
			> tmp_data/gop_debates/speakers
	mkdir -p tmp_data/gop_debates/data
	while read speaker; do \
			find tmp_data/gop_debates/raw -type f -exec cat {} \; \
				| grep "^$$speaker" \
				| sed "s/^$$speaker//g" \
				> tmp_data/gop_debates/data/$$speaker; \
	done < tmp_data/gop_debates/speakers

gop_debates.cleaned: gop_debates
	mkdir -p tmp_data/gop_debates/cleaned
	while read speaker; do \
		./bin/preprocess_text tmp_data/gop_debates/data/$$speaker \
			>  tmp_data/gop_debates/cleaned/$$speaker; \
	done < tmp_data/gop_debates/speakers

################################################################################
# Political campaign speech transcripts
################################################################################
download_transcripts:
	# youtube-dl outputs all kinds of error codes, even when it does what it should
	# this causes the Makrfile from continuing to the next command, so we have to force
	# exit code zero ...
	# right side broadcasting
	mkdir -p tmp_data/transcripts
	cd tmp_data/transcripts && youtube-dl -i --yes-playlist \
		--write-auto-sub --write-sub --sub-lang en --skip-download \
		https://www.youtube.com/playlist?list=PLuXXbBFpPc0lb4_FdI1NOPTCAYroXOywl \
		|| exit 0
	# donald trump speeches and rallies
	cd tmp_data/transcripts && youtube-dl -i --yes-playlist \
		--write-auto-sub --write-sub --sub-lang en --skip-download \
		https://www.youtube.com/playlist?list=PL-NSU9cjYpaHyHPwhvWYq0ooFnuz7f-G0 \
		|| exit 0
	# trump tv network - thank you rally
	cd tmp_data/transcripts && youtube-dl -i --yes-playlist \
		--write-auto-sub --write-sub --sub-lang en --skip-download \
		https://www.youtube.com/playlist?list=PLbBZQ-qTn7spy0HwpLl4qFTs5Vm1f-6Uq \
		|| exit 0
	# RBC NETWORK BROADCASTING - Donald Trump Speeches & Events
	cd tmp_data/transcripts && youtube-dl -i --yes-playlist \
		--write-auto-sub --write-sub --sub-lang en --skip-download \
		https://www.youtube.com/playlist?list=PLocvq02h-FETPLh3YW5TK5QNbAYkmLKKG \
		|| exit 0

transcript_srts: download_transcripts
	find tmp_data/transcripts/ -name '*.vtt' -exec ffmpeg -n -i '{}' '{}.srt' \;

fulltext.titles: transcript_srts
	find tmp_data/transcripts/ -iname '*.srt' -type f > tmp_data/fulltext.titles

fulltext: fulltext.titles
	# make sure we're using the same index across
	# then use it here for clustering pipe and also later
	cat tmp_data/fulltext.titles | xargs -d '\n' -I '{}' ./bin/srt2text '{}' > tmp_data/fulltext

fulltext.cleaned:
	cat tmp_data/fulltext | ./bin/preprocess_text > tmp_data/fulltext.cleaned

################################################################################
# VOX pipeline
################################################################################

vox_corpus.titles:
	cat tmp_data/elenadata-vox-articles/dsjVoxArticles.tsv \
		| awk -F'\t' '{print $$4}' \
		| tail -n +2 \
		| awk '{print $$1}' \
		| awk -F'-' '{print $$2, "-", $$3, "-", $$1}' | tr -d ' ' \
		> tmp_data/vox_corpus.titles

# NOTE: This data is available from data.world
vox_corpus: vox_corpus.titles
	mkdir -p ./tmp_data/vox_documents
	find tmp_data/vox_documents/ -iname '*.txt*' -delete
	split -l 1 tmp_data/elenadata-vox-articles/dsjVoxArticles.tsv ./tmp_data/vox_documents/
	# remove the header file
	rm -f ./tmp_data/vox_documents/aa
	find ./tmp_data/vox_documents/ -not -iname '*.txt' -exec \
	        html2text -width 8000 -nometa -ascii -nobs -o {}.txt {} \;
	# eliminate line breaks per-document
	find ./tmp_data/vox_documents/ -iname '*.txt' \
		-exec sh -c "cat {} | tr '\n' ' ' | sed 's/\\n//g' > {}.nobr" \;
	# consolidate each document into a single file, one per line
	find ./tmp_data/vox_documents/ -iname '*.nobr' -exec sh -c "cat {} && echo" \; \
		> ./tmp_data/vox_corpus

vox_corpus.cleaned: vox_corpus
	./bin/preprocess_text ./tmp_data/vox_corpus > ./tmp_data/vox_corpus.cleaned

################################################################################
# Generic extraction, embedding and clustering flow, make sure to set CORPUS
################################################################################
# Keyword extraction
keywords:
	./bin/tf-idf-extract.py tmp_data/${CORPUS}.cleaned \
		--pct 0.75 \
		> tmp_data/${CORPUS}.keywords

# Word embeddings
# NOTE: fasttest -thread 1 makes the results deterministic due to randomness
# in mult-threaded async gradient descent algorithm
# -loss softmax makes the training take REALLY long & aren't as good
keywords.vec:
	# NOTE: cbow works well with smaller datasets like the political speech corpus
	if [[ ${EMBED_PROGRAM} == "fasttext" ]]; then \
		echo Using fasttext; \
		fasttext ${EMBEDTYPE} -thread 1 \
			-minCount 0 \
			-ws 10 \
			-dim 100 -input tmp_data/${CORPUS}.cleaned \
			-output tmp_data/${CORPUS}.${EMBED_PROGRAM}.model; \
		fasttext print-vectors tmp_data/${CORPUS}.${EMBED_PROGRAM}.model.bin \
			< tmp_data/${CORPUS}.keywords \
			> tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.vec; \
	elif [[ ${EMBED_PROGRAM} == "word2vec" ]]; then \
		echo using word2vec; \
		word2vec -thread 1 \
			-min-count 0 \
			-cbow 0 \
			-window 10 \
			-size 100 -train tmp_data/${CORPUS}.cleaned \
			-output tmp_data/${CORPUS}.${EMBED_PROGRAM}.model.vec; \
		./bin/words2vectors.py tmp_data/${CORPUS}.${EMBED_PROGRAM}.model.vec \
			< tmp_data/${CORPUS}.keywords \
			> tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.vec; \
	fi
	echo "Word embeddings created"

keywords.uniq.vec: keywords.vec
	cat tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.vec \
		| sort -n \
		| uniq \
		> tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.uniq.vec

clusters: keywords.uniq.vec
	echo ${CLUSTERS_DIR}
	mkdir -p tmp_data/${CLUSTERS_DIR}
	rm -f tmp_data/${CLUSTERS_DIR}/*
	# use the non-uniq vectors so we capture the distribution across documents of keywords
	./bin/kmeans.py \
		--distance ${DISTANCE} \
		--algorithm ${K_MEANS_VARIANT} \
		./tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.uniq.vec \
		./tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.uniq.vec \
		${N_CLUSTERS} \
		tmp_data/${CLUSTERS_DIR} \
		> tmp_data/${CLUSTERS_DIR}/output.log
	wc -l tmp_data/${CLUSTERS_DIR}/output.log

timeseries: clusters presentation_circles
	mkdir -p plot
	rm -f plot/${CORPUS}*
	cat data/ranked_cluster_evaluations_combined.txt \
		| awk -F'(' '{print $2}' \
		| awk -F')' '{print $1}' \
		> tmp_data/all_clusters.txt
	# optimally computer the cluster-bigram timeseries
	./bin/timeseries_preprocess.py \
		./tmp_data/all_clusters.txt \
		./tmp_data/fulltext.cleaned \
		./tmp_data/fulltext.titles \
		> ./tmp_data/ts_output.json

################################################################################
# EVALUATION
################################################################################
build_gram_freqs:
	# cat tmp_data/vox_corpus.cleaned \
	#| ./bin/evaulate_preprocess.py \
	#	tmp_data/evaluation/vox_unigrams.json \
	#	tmp_data/evaluation/vox_bigrams.json
	cat tmp_data/fulltext.cleaned \
	| ./bin/evaulate_preprocess.py \
		tmp_data/evaluation/fulltext_unigrams.json \
		tmp_data/evaluation/fulltext_bigrams.json

evaluate:
	./bin/evaluate.py \
		tmp_data/evaluation/fulltext_unigrams.json \
		tmp_data/evaluation/fulltext_bigrams.json \
		tmp_data/${CLUSTERS_DIR}/clusters.csv \
	> tmp_data/evaluation/${CLUSTERS_DIR}_evaluation.json

full_evaluation:
	#./bin/evaluate_k.sh cluster
	./bin/evaluate_k.sh eval
	./bin/plot_evaluation.py tmp_data/evaluation plot/evaluation

################################################################################
# PRESENTATION DATA
################################################################################
presentation_circles:
	./bin/eval_pprint.py 100..1000..10 --output combined \
		> data/ranked_cluster_evaluations_combined.txt
	./bin/eval_pprint.py 100..1000..10 --output all \
		> data/ranked_cluster_evaluations.txt
	mkdir -p plot/presentation
	for T in word2vec_spherical word2vec_kmeans fasttext_spherical fasttext_kmeans lda_lda; do \
		cat data/ranked_cluster_evaluations.txt \
			| awk 'BEGIN{X=0}; {X=X+1; print X, $$1, NF-2, $$2}; END{print $$X}' \
			| grep $$T \
			> plot/presentation/$$T.dat; \
	done

presenatation_ranked:
	#./bin/eval_pprint.py 100..1000..10 --output combined \
	#    > data/ranked_cluster_evaluations_combined.txt
	#./bin/eval_pprint.py 100..1000..10 --output all \
	#    > data/ranked_cluster_evaluations.txt
	cat data/ranked_cluster_evaluations.txt \
		| awk 'BEGIN{X=0}; {X=X+5; print X, $$1, $$2}; END{print $$X}' \
		> tmp_data/all_clusters.dat
	#cat tmp_data/all_clusters.dat \
	#	| sed 's/\(.*\)\-[0-9]*/\1/g' \
	#	| awk '{$$1=""; print $$0}' \
	#	| grep -v -e '^$$' \
	#	| awk 'BEGIN{X=0}; {X=X+1; print X, $$0}' \
	#	> tmp_data/all_clusters_uniq.dat
	#for T in word2vec_kmeans word2vec_spherical fasttext_kmeans fasttext_spherical lda_lda; do \
	#	cat tmp_data/all_clusters_uniq.dat \
	#			| grep $$T \
	#			> plot/presentation/$$T.dat; \
	#done
	for T in word2vec_kmeans word2vec_spherical fasttext_kmeans fasttext_spherical lda_lda; do \
		cat tmp_data/all_clusters.dat \
			| grep $$T \
			> plot/presentation/$$T.dat; \

################################################################################
# Utils
################################################################################
plot_k_wssse:
	./bin/kmeans-variance.py ./tmp_data/${CORPUS}.${EMBED_PROGRAM}.keywords.uniq.vec \
		--max_k 2000 \
		--min_k 1 \
		--step 5 \
		--algorithm ${K_MEANS_VARIANT}
		#> data/${CORPUS}_wssse_${K_MEANS_VARIANT}.dat

files2dates:
	find tmp_data/transcripts/ -iname '*.srt' -type f -regex '.*[0-9]+.[0-9]+.[0-9]+.*' \
		| sed 's/.*[( ]\([0-9]\+\).\([0-9]\+\).\([0-9]\+\).*/\1 \2 \3/g' \
		| grep -v \.en\.

mallet_benchmark:
	mallet import-file --input ./tmp_data/${CORPUS}.cleaned --output ./tmp_data/${CORPUS}.mallet --keep-sequence --remove-stopwords
	mallet train-topics --show-topics-interval 1000 --input ./tmp_data/${CORPUS}.mallet --num-topics ${N_CLUSTERS} --output-state ./tmp_data/${CORPUS}.topics.gz
