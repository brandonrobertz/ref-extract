#!/usr/bin/env python
from __future__ import print_function
import sys
import re

import numpy as np


def logerr(msg):
    sys.stderr.write(msg)
    sys.stderr.write('\n')


with open(sys.argv[1], 'r') as f:
    data = f.readlines()

word_vectors = {}
samples, dim = data[0].split()

for line in data[1:]:
    word, vec = line.split(' ', 1)
    word_vectors[word] = np.array([
        float(i) for i in vec.split()
    ], dtype='float32')

logerr("word_vectors (keys) %s" % word_vectors.keys())

for line in sys.stdin.readlines():
    line = line.strip()
    if not line:
        # print "Skipping blank line"
        continue

    words = re.split('\s+', line)
    for word in words:
        if word not in word_vectors:
            logerr("Word not found %s" % word)
            continue

        vector = word_vectors[word]
        string_rep = " ".join(map(lambda x: str(x), vector.tolist()))
        print("%s %s" % (word, string_rep))

