#!/bin/bash

cat *psd_histo > histo

awk 'NR>11 {print}' histo | awk -F' ' '{print $1}' > binn
awk 'NR>11 {print}' histo | awk -F' ' '{print $4}' > dd

#Pasting to a new csv file
echo -e "bin, derivative_dist" > Finaldata.csv
paste binn dd -d"," >> Finaldata.csv
