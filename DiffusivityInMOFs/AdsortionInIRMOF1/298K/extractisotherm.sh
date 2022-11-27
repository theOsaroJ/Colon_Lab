#!/bin/bash

grep -F 'Average loading absolute [molecules/unit cell]' *.data > sampleAd.txt
grep -F 'Partial pressure' *.data> samplePr.txt

#Awking the pipe to get values
awk -F' ' '{print $4}' samplePr.txt > Pres
awk -F' ' '{print $7}' sampleAd.txt > upt
awk -F' ' '{print $9}' sampleAd.txt > err


#Pasting to a new csv file
echo -e "Pressure,Uptake,Error" > Finaldata.csv
paste Pres upt err -d"," >> Finaldata.csv

sort -n Finaldata.csv > CompleteData.csv
