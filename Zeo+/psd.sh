#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 8
#$ -N PSD
#$ -t 1-12

# The MOFs cif files
cifs=(Cu-BTC.cif NU-800.cif DUT-32.cif DUT-49.cif IRMOF-1.cif IRMOF-10.cif IRMOF-16.cif UIO-66.cif MgMOF-74.cif ZIF-8.cif PCN-61.cif MOF-177.cif)

# Creating a dir for each cif file
dir=DIR.${cifs[ $SGE_TASK_ID - 1]}
mkdir $dir

# copying network + job particular cif file into its dir.
cp network plot.py ${cifs[ $SGE_TASK_ID - 1]} $dir
cd $dir

# running the job from each directory.

./network -ha -psd 1.4 1.4 50000 ${cifs[ $SGE_TASK_ID - 1]}

# extracting bin and derivative_dist from results
cat *psd_histo > histo

awk 'NR>11 {print}' histo | awk -F' ' '{print $1}' > binn
awk 'NR>11 {print}' histo | awk -F' ' '{print $4}' > dd

#Pasting to a new csv file
echo -e "bin, derivative_dist" > Finaldata.csv
paste binn dd -d"," >> Finaldata.csv
rm binn dd histo

#Plotting the results
module load python

python3 plot.py

module unload python
cd ../
