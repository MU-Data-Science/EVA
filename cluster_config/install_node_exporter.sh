#!/bin/bash

# Downloads the node exporter, extracts the executable, and starts the node_exporter process.
if !(test -f "node_exporter-1.6.1.linux-amd64.tar.gz"); then
    wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
    tar xvzf node_exporter-1.6.1.linux-amd64.tar.gz
    cd node_exporter-1.6.1.linux-amd64
fi


# Run node exporter in a detached screen session named 'node'
screen -S node -d -m ./node_exporter
