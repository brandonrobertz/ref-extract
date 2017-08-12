#!/bin/bash
OUTDIR=${1}
INPUT=${2}

if [ -z ${OUTDIR} ]; then
  echo USAGE: scrape_gop_debates.sh DIR_TO_WRITE INPUT_LIST_LOCATION
  echo Error: missing write directory
  exit 1
fi


if [ -z ${INPUT} ]; then
  echo USAGE: scrape_gop_debates.sh DIR_TO_WRITE INPUT_LIST_LOCATION
  echo Error: input list location
  exit 1
fi


if [ -d ${OUTDIR} ]; then
  echo Cleaning gop_debates directory
  rm -rf ${OUTDIR}/*
fi

# download and extract debate transcript from list
while read data; do
  DATE=`echo $data | awk -F'|' '{print $1}' | sed 's/^\(.*\) \([0-9]\+\)[thrds]\+, \(.*\)$/\1 \2 \3/g' | xargs -I{} date -d {} +%Y-%m-%d`
  LOCATION=`echo $data | awk -F'|' '{print $2}'`
  URL=`echo $data | awk -F'|' '{print $3}'`

  echo Extracting $URL to $DATE

  # we use tail because xidel is outputting "false" as
  # the first line during extraction
  curl -s $URL \
    | ./bin/xidel -s --data=- -e --input-format=html \
      --output-format=adhoc \
      --xpath3='//span[@class="displaytext"]/p' \
    | tail -n +2 \
    | ./bin/extract_speakers.py \
    > ${OUTDIR}/$DATE

done < ${INPUT}
