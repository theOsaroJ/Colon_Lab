#!/bin/bash

#$ -q hpc@@colon
#$ -N RQNS
#$ -pe smp 1


#### ------- Objective ------- ####
## To run Alchemical OT1_1S in OPT1  MOF simulations until we get a max. relative error within 2%

#Making a datafile
data="Prior.csv"
#UUU=$(tail -n 1 Prior.csv)

#Initialising a variable called one
One=1

# Declaring the number of samples present in the prior sample
N_Samp=$(wc -l < Prior.csv)

#Removing 1 from N_Samp, since the first row was coloumn names
N_Samp=$[ $N_Samp - $One ]

##creating error files and populating the header line at the top

#checking if the files exist or not and then removing them
if [ -f rel_true.csv ]; then
rm rel_true.csv
fi

if [ -f rel.csv ]; then
rm rel.csv
fi

if [ -f mean.csv ]; then
rm mean.csv
fi

#creating those files
touch rel_true.csv mean.csv rel.csv

#populating the top row
for ((i=1;i<=279;i++))
do
        echo -n "${i}," >> rel_true.csv
        echo -n "${i}," >> rel.csv
done

#Last row without the comma at the end
for ((i=280;i<=280;i++))
do
        echo -n "${i}" >> rel_true.csv
        echo -n "${i}" >> rel.csv
done

echo "GP-based_rel_Error,RRMSE" >> mean.csv

#going to the next line in these files
echo " " >> rel_true.csv
echo " " >> rel.csv


##### -----------   TESTING    -----------
#Testing N_samp --- generally test lines should be commented out when the code is running well
#echo $N_Samp
#### ------------ TESTING DONE -----------

#Declaring a string variable Fin which would compare if the active learning was successful or not
Fin="NOT_DONE"

#for loop is finished
#Going for the max number of loops, in case if loop inside our for does not stop
for ((i=1;i<=280;i++))
do

        #creating output for python
        if [[ -f output ]]; then
        rm output
        touch output
        fi

        #Loading modules
        module load python

        # funneling the python output to output
        python3 GP.py > output

        Fugacity=$(awk 'FNR==1 {print $1}' output)
        Epsilon=$(awk 'FNR==1 {print $2}' output)
        Siggma=$(awk 'FNR==1 {print $3}' output)
        Uptake=$(awk 'FNR==1 {print $4}' output)
        Flag=$(awk 'FNR==2 {print $1}' output)
        SSS="$Fugacity,$Epsilon,$Siggma,$Uptake"

        #going to the next line in these files
        echo " " >> rel_true.csv
        echo " " >> rel.csv
        echo " " >> mean.csv

        #Removing brackets in the mean.csv from rrmse output from error_estimator code
        sed -i 's/[][]//' mean.csv
        sed -i 's/[][]//' mean.csv

        #converting Index from a string to integer
        Index=${Index#0}
        #unloading python 3.7 (the latest version) since RASPA is incompatible with Python 3.7
        module unload python

        ##Checking if the uncertainty (sigma) is lower than the limit; if not we need to do more simulations
        if [ $Flag == $Fin ];
        then
                # Printing whether the code has converged or not, and the index with max. uncertainty
                echo "Active learning still not finished!"
                echo $Index
                if [ $SSS = $UUU ];
                then
                        break

                else
                        echo "$Fugacity,$Epsilon,$Siggma,$Uptake" >> ${data}
                        UUU=$(tail -n 1 Prior.csv)
        #               rm output2
        #               rm output3
                fi
                # removing the sample file
        else
                #In case If doesn't satisfy, (which means the uncertainty is lower than 2% for all points), break out of this loop and finish Active learning, the model is ready
                break
        fi
done
