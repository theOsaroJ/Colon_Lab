#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 8
#$ -N PSD
#$ -t 1-3

# The MOFs cif files
cifs=(Cu-BTC.cif NU-800.cif DUT-32.cif)

# Creating a dir for each cif file
dir=DIR.${cifs[ $SGE_TASK_ID - 1]}
mkdir $dir

# copying network + job particular cif file into its dir.
cp network ${cifs[ $SGE_TASK_ID - 1]} $dir
cd $dir

# running the job from each directory.

./network -ha -psd 1.4 1.4 10 ${cifs[ $SGE_TASK_ID - 1]}

# extracting bin and derivative_dist from results
cat *psd_histo > histo

awk 'NR>11 {print}' histo | awk -F' ' '{print $1}' > binn
awk 'NR>11 {print}' histo | awk -F' ' '{print $4}' > dd

#Pasting to a new csv file
echo -e "bin, derivative_dist" > Finaldata.csv
paste binn dd -d"," >> Finaldata.csv
rm binn dd histo
cd ../
