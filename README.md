# Ref Extract

![Ref Extract](../blob/master/public_html/img/red-extract.gif?raw=true)

Reference extraction from noisy data using k-means clustering of word embeddings. Implements model described in _Extracting references from political speech auto-transcripts_ ([pdf](https://drive.google.com/open?id=0B8CcT_0LwJ8QaUtYZ1R6c1FqR2M)). This is intended not as a general ease-of-access tool (yet), but as a technical demonstration of the technique described in the paper.

# Usage (Demo)

To simply run the demo run the following `make` commands:

    make clusters

This will run the embedding and clustering routines. You need to have [fastText](https://github.com/facebookresearch/fastText) on your path somewhere and have all the python dependencies (run `pip install -r requirements.txt` to install them in to your virtualenv).

Everything can be ran from the Makefile using the sample dataset. You can download the whole dataset using the command:

    make fulltext fulltext.cleaned

If you want to use your own dataset, place your data in `tmp_dir/fulltext`. The `tmp_dir` has been created, simply format your data as one-line per document. To pre-process your data run:

    make fulltext.cleaned

This will leave a pre-processed file in `tmp_dir/fulltext.cleaned`.

