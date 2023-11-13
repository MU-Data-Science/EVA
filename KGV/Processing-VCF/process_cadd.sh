#!/bin/bash
# ./process_cadd.sh /path/to/Variant_Analysis_Output_XXX_XX_2021_hg19/CADD

set -e

dir_path=$(dirname $0)

echo "Starting unzipping"
DIR=$1/unzipped
mkdir -p $DIR
for f in $(find $1 -type f -name "*.gz")
do
    t=$(basename $f .gz)
    gunzip -c $f > $DIR/$t
    echo $DIR/$t
done

echo "Processing CADD files"
N3_DIR=$1/N3
python3 $dir_path/ParseTSV.py -i $DIR -o $N3_DIR 
