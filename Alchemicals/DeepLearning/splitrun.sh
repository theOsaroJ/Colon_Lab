#!/bin/bash
#$ -q hpc@@colon
#$ -pe smp 8
#$ -N split

module load python
python3 split.py

module unload python

#Creating a folder and putting each file into folder
for f in OPT*.csv
do
  subdir=${f%%.*}
  [ ! -d "$subdir" ] && mkdir -- "$subdir"
  mv -- "$f" "$subdir"
done

for d in *; do
  if [ -d "$d" ]; then
    ( cd "$d" &&
#Creating a copy of file to replace fugacity,eps,sig,charge and bondlength with placehold

head -n 2 OPT*.csv > one.csv

#picking the placeholders index now
vvv=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $28}')
www=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $29}')
xxx=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $30}')
yyy=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $31}')
zzz=$(awk 'FNR==2 {print $1}' one.csv | awk -F',' '{print $32}')

#Creating a new file to delete and paste later.
cat one.csv > two.csv
cut -d, -f 1-27 two.csv > three.csv

#passing the values to a file
echo "Fugacity,chg,bond_length,eps_eff,sig_eff" > four.csv
echo "$vvv,$www,$xxx,$yyy,$zzz" >> four.csv

#replacing the indexes by placeholders index with placeholders
sed -i 's/'${vvv}'/AAA/' four.csv
sed -i 's/'${www}'/BBB/' four.csv
sed -i 's/'${xxx}'/CCC/' four.csv
sed -i 's/'${yyy}'/DDD/' four.csv
sed -i 's/'${zzz}'/EEE/' four.csv

#appending file
paste three.csv four.csv -d','> DL_AL.csv

#converting name of main files
mv OPT*.csv CompleteData.csv

#removing unwanted files
rm one.csv two.csv three.csv four.csv
 )
  fi
done
