#!/bin/bash
# ./process_vcfs.sh /path/to/VCF

set -e
#full_path=$(realpath $0)
dir_path=$(dirname $0)

echo "Starting unzipping"
DIR=$1/unzipped
ANNOTATED_DIR=$1/annotated
mkdir -p $DIR
for f in $(find $1 -type f -name "*.gz")
do
    t=$(basename $f .gz)
    gunzip -c $f > $DIR/$t
    echo $DIR/$t
done

echo "Processing files"
python3 $dir_path/ProcessFiles.py -i $DIR -o $ANNOTATED_DIR -j /path/to/snpEff/snpEff.jar

echo "Annotating N3 with vcf2rdf tool"
for file in $(find $ANNOTATED_DIR -type f -name "*.vcf")
do 
    
    echo $file
    /path/to/sparqling-genomics-0.99.11/tools/vcf2rdf/vcf2rdf -i $file > ${file/.vcf/.n3}
done

N3_DIR=$ANNOTATED_DIR/N3
mkdir -p $N3_DIR
for file in $(find $ANNOTATED_DIR -type f -name "*.n3")
do
    mv $file $N3_DIR
done

echo "Annotating N3 to generate NQ"
python3 $dir_path/AnnotateGraphName.py -e $1/../RNA-Sequence-Details.xlsx -n $N3_DIR

NQ_DIR=$ANNOTATED_DIR/NQ
mkdir -p $NQ_DIR
for file in $(find $N3_DIR -type f -name "*.nq")
do
    mv $file $NQ_DIR
done
