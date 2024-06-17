#!/bin/bash

# Copies data from SOURCE_BUCKET -> TMP_BUCKET -> DEST_BUCKET.
# This is useful for reducing cost to Ludwig's public bucket by ensuring copies
# from Ludwig's buckets always happen within region.

if [[ "$#" != 3 ]] ; then
    echo "Usage: $0 <dcnlp-hub-path> <tmp-path> <our-path>"
    exit 1
fi

# Remove trailing slashes if they exist
dcnlp_path=${1%/}
tmp_path=${2%/}
our_path=${3%/}

echo "Counting files"
num_files=$(aws s3 ls ${dcnlp_path}/ | tqdm | wc -l)
echo "Found ${num_files} files"

aws s3 ls ${dcnlp_path}/ | \
    awk '{print $4}' | \
    xargs -I {} -P 60 aws s3 cp ${dcnlp_path}/{} ${tmp_path}/{} | \
    tqdm --total ${num_files} >/dev/null

aws s3 ls ${tmp_path}/ | \
    awk '{print $4}' | \
    xargs -I {} -P 60 aws s3 cp ${tmp_path}/{} ${our_path}/{} | \
    tqdm --total ${num_files} >/dev/null

