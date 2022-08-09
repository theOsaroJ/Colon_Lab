#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 8
#$ -N split

#Entering into each MOF folder to execute a command after copying split python file
for d in *; do
  if [ -d "$d" ]; then
    ( cp buildprior.py "$d" &&
cd "$d"
#taking columns that matter
cat CompleteData.csv > one.csv
cut -d, -f 28-33 one.csv > two.csv

module load python
python3 buildprior.py

#deleting the header
sed -i '1d' 1*csv 3*csv 5*csv 7*csv

#deleting everything except the first and last line
sed -i -n '1p;$p' 1*csv 3*csv 5*csv 7*csv

#Creating the Prior now
echo "Fugacity,chg,bond_length,eps_eff,sig_eff,prediciton" > three.csv
cat 1*csv 3*csv 5*csv 7*csv >> three.csv

sort -n three.csv > Prior.csv
#Removing the annoying ^M character
sed -i "s/\r//g" Prior.csv

rm 1*csv 3*csv 5*csv 7*csv one.csv two.csv three.csv
 )
  fi
done
