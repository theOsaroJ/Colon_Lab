#!/bin/bash
#$ -q hpc@@colon
#$ -N methane_100
#$ -pe smp 8

##------------------------------loading lammps---------------------------##
module load lammps

##-------------------------running the nvt job---------------------------##
mpirun -np 8 lmp_mpi -in in_nvt

##------------------------running PyLAT-----------------------------------##
sh compile.sh
python3 PyLAT.py -d -g --mol ch4 --nummol 10 -p ./ -f msd.json -v 2 mol.log combine.data  mol.molecs.lammpstrj

##-----------------Extracting diffusivity from pylat and analyze.py scripts---------------##
##run analyze.py
if [[ -f values.txt ]]; then
rm values.txt
fi

#python3 analyze.py > values.txt
grep 'ch4' msd.json | head -n 1 | awk -F ' ' '{print $2}' | sed 's/.$//' >> values.txt

##-------------------Plotting RDFs and MSD from Pylat json file--------------------------##
#python3 plot.py
