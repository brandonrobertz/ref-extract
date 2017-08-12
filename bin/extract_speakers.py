#!/usr/bin/env python2
""" Read speech transcripts from stdin and separate & collect speakers' text
"""
import re
import sys

speakers = {}

def splittext(line):
    speaker = None
    said = None

    matches = re.findall(r'^([A-Z\-\s]+):(.*)$', line)

    # we have a text line without speaker (continuation)
    if not matches:
        return None, line

    if len(matches) > 1:
        raise RuntimeError('Strange extraction on line: {}'.format(line))

    speaker = matches[0][0]
    said = matches[0][1].strip()

    return speaker, said

lastspeaker = None
for line in sys.stdin.readlines():
    line = line.strip()
    if not line:
        # print "Skipping blank line"
        continue

    speaker, text = splittext(line.strip())

    if speaker and speaker not in speakers:
        speakers[speaker] = ''

    # we can't use text without context
    if not speaker and not lastspeaker:
        sys.stderr.write("First line, no speaker found\n")
        continue
    # we have a new speaker
    elif speaker and lastspeaker != speaker:
        lastspeaker = speaker
    # save new/last speaker's words
    speakers[lastspeaker] += text

for speaker in speakers:
    nobreaks = speakers[speaker].replace('\r', ' ').replace('\n', ' ')
    cleaned = re.sub('\s+', ' ', nobreaks)
    print speaker, cleaned
