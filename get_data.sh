#!/usr/bin/env bash

mkdir -p data
cd data
mkdir -p aol
cd aol

echo "Downloading AOL"
wget --quiet --continue http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/aol-data.tar.gz
tar -zxf aol-data.tar.gz
rm aol-data.tar.gz
mv AOL-user-ct-collection org
cd org
gzip -d *.gz
cd ../../..

echo "Happy language modeling and query auto-completion :)"