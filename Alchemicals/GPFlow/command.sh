#!/bin/bash

for d in $(cat label.txt)
do
cd $d
cd RBF

cp ../../tmp.sh .
sed -i 's/INDEX/'$d'/' tmp.sh
qsub tmp.sh
cd ../../
done
