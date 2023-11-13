#!/bin/bash

echo "Removing old JNL file.."
rm /mydata/dgl/blazegraph.jnl

find $1 -name "VCF" -print0 | while read -d $'\0' file
do
    echo "Loading VCF files into Blazegraph.."
    java -Xmx4g -cp /mydata/dgl/blazegraph.jar com.bigdata.rdf.store.DataLoader -verbose -namespace kb /mydata/dgl/quad.properties $file
done

find $1 -name "CADD_Scores" -print0 | while read -d $'\0' file
do
    echo "Loading CADD_Scores files into Blazegraph.."
    java -Xmx4g -cp /mydata/dgl/blazegraph.jar com.bigdata.rdf.store.DataLoader -verbose -defaultGraph http://sg.org -namespace kb /mydata/dgl/triple.properties $file
done

echo "Starting up Blazegraph.."
java -server -Xmx4g -jar /mydata/dgl/blazegraph.jar