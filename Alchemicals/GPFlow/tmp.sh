!/bin/bash
#$ -q hpc@@colon
#$ -N NewGPF
#$ -pe smp 4

module load python
##--------------------------Using UGE for /tmp jobs----------------------------##

##--------------------Deciding to cal. each mof R_squared( take just adsorption)---------------------##
awk -F',' '{print $33}' CompleteData.csv > actual

# Set the paths
export WORKDIR=$PWD    # $PWD - means present working directory, but you can specify any other directory path
export PARENT=/scratch365/eosaro/Research/AlchemicalCollaboration/FESQL/Deeplearning/Normal
export TMPDIR=/tmp/tmp_eosaroNG_INDEX

# Create temp directory
#rm -rf tmp/tmp_eosaro_test
mkdir ${TMPDIR}
cd ${TMPDIR}
cp -r ${WORKDIR}/* ${TMPDIR}

#Copying the files needed for the simulations
cp ${PARENT}/gas_loading_prediction.h5 .
cp ${PARENT}/model.py .
cp ${PARENT}/training.csv .

#cd RBF
#### ------- Objective ------- ####
## To run Alchemical OT1_1S in OPT1  MOF simulations until we get a max. relative error within 2%

#Making a datafile
data="Prior.csv"

#Initialising a variable called one
One=1

# Declaring the number of samples present in the prior sample
N_Samp=$(wc -l < Prior.csv)

#Removing 1 from N_Samp, since the first row was coloumn names
N_Samp=$[ $N_Samp - $One ]

##creating error files and populating the header line at the top

#populating the top row
for ((i=1;i<=2799;i++))
do
        echo -n "${i}," >> rel_true.csv
        echo -n "${i}," >> rel.csv
done

#Last row without the comma at the end
for ((i=2800;i<=2800;i++))
do
        echo -n "${i}" >> rel_true.csv
        echo -n "${i}" >> rel.csv
done

echo "GP-based_rel_Error,Abs_Mean_Error,RRMSE" >> mean.csv

#going to the next line in these files
echo " " >> rel_true.csv
echo " " >> rel.csv

#Declaring a string variable Fin which would compare if the active learning was successful or not
Fin="NOT_DONE"

#for loop is finished
#Going for the max number of loops, in case if loop inside our for does not stop
for ((i=1;i<=2800;i++))
do

        #creating output for python
        if [[ -f output ]]; then
        rm output
        touch output
        fi

        # funneling the python output to output
        python3 GP.py > output
        #Taking the GP relative error and sending to document
        som=$(awk 'FNR==3 {print $1}' output)
        echo $som >> cummulative.csv

        ##----------------Pasting actual & predicted into its unique csv file-------------------##
        paste actual pred.csv -d"," > r_sq.csv

        ##---------Removing the annoying ^M character-------------###
        sed -i "s/\r//g" r_sq.csv

        ##---------------Computing the R_sq at this time----------###
        python3 plot_individuals.py > r_sq.txt
        rr_sq=$(cat r_sq.txt)
        echo $rr_sq >> squares.csv

        #Initialising variables that will store the array Index for max. uncertainty, and the flag which tells if the code has converged or not
        Fugacity=$(awk 'FNR==1 {print $1}' output)
        Charge=$(awk 'FNR==1 {print $2}' output)
        BL=$(awk 'FNR==1 {print $3}' output)
        Epsilon=$(awk 'FNR==1 {print $4}' output)
        Siggma=$(awk 'FNR==1 {print $5}' output)
        Flag=$(awk 'FNR==2 {print $1}' output)

        ##### -----------   TESTING    -----------
        #echo $Index,$Max
        #Flag=${lim#0}
        #echo $Flag
        #removing the output file
        #rm output
        #### ------------ TESTING DONE -----------

        #### --------- Running a script to extract RRMSE and Rel. Error ---------- ####

        #python error_estimator.py

        #going to the next line in these files
        echo " " >> rel_true.csv
        echo " " >> rel.csv
        echo " " >> mean.csv

        #Removing brackets in the mean.csv from rrmse output from error_estimator code
        sed -i 's/[][]//' mean.csv
        sed -i 's/[][]//' mean.csv

        #### --------- Error extraction is completed  --------- ####

        #converting Index from a string to integer
        Index=${Index#0}
        #### -------- Creating a Variable to replace fugacity, epsilon and sigma in DL_AL.csv file ----------###
        VVV=$(grep 'AAA' DL_AL.csv | awk -F ',' '{print $28}')
        WWW=$(grep 'BBB' DL_AL.csv | awk -F ',' '{print $29}')
        XXX=$(grep 'CCC' DL_AL.csv | awk -F ',' '{print $30}')
        YYY=$(grep 'DDD' DL_AL.csv | awk -F ',' '{print $31}')
        ZZZ=$(grep 'EEE' DL_AL.csv | awk -F ',' '{print $32}')
        #unloading python 3.7 (the latest version) since RASPA is incompatible with Python 3.7
        #module unload python

        ##Checking if the uncertainty (sigma) is lower than the limit; if not we need to do more simulations
        if [[ $Flag == $Fin ]];
        then
                # Printing whether the code has converged or not, and the index with max. uncertainty
                                echo "Active learning still not finished!"
                echo $Index

                #Adding the next pressure simulation point
                N_Samp=$[ $N_Samp + $One ]

                #### ---------- Preparing for submitting the next simulation ---------- ####
                #making the directory for next simulation step
                mkdir $N_Samp
                #copying the input file from the principal folder
                cp gas_loading_prediction.h5 model.py training.csv $N_Samp
                cp DL_AL.csv $N_Samp
                cd $N_Samp


                #changing the placeholder to Fugacity, Epsilon and Sigma from Output
                sed -i 's/'${VVV}'/'${Fugacity}'/' DL_AL.csv
                sed -i 's/'${WWW}'/'${Charge}'/' DL_AL.csv
                sed -i 's/'${XXX}'/'${BL}'/' DL_AL.csv
                sed -i 's/'${YYY}'/'${Epsilon}'/' DL_AL.csv
                sed -i 's/'${ZZZ}'/'${Siggma}'/' DL_AL.csv

                #running the prediction of the loading using the DL model
                python3 model.py
                #collecting the loadings from the csv file
                awk -F',' '{print $2}' predictions.csv > loadings.csv

                #Removing the brackets from the values
                sed -i 's/[][]//' loadings.csv
                sed -i 's/[][]//' loadings.csv

                #changing any negative value of loading to 0.000001
                sed -i 's/^-0.*/0.0000001/g' loadings.csv
                sed -i 's/^-1.*/0.0000001/g' loadings.csv
                sed -i 's/^-2.*/0.0000001/g' loadings.csv
                sed -i 's/^-3.*/0.0000001/g' loadings.csv
                sed -i 's/^-4.*/0.0000001/g' loadings.csv
                sed -i 's/^-5.*/0.0000001/g' loadings.csv
                sed -i 's/^-6.*/0.0000001/g' loadings.csv
                sed -i 's/^-7.*/0.0000001/g' loadings.csv
                sed -i 's/^-8.*/0.0000001/g' loadings.csv
                sed -i 's/^-9.*/0.0000001/g' loadings.csv

                Uptake=$(awk 'FNR==2 {print $1}' loadings.csv)
                echo "$Fugacity,$Charge,$BL,$Epsilon,$Siggma,$Uptake" >> ../${data}

                #Removing heavy storage files
                rm training.csv
                cd ../
        else
                #In case If doesn't satisfy, (which means the uncertainty is lower than 2% for all points), break out of this loop and finish Active learning, the model is ready
                break
        fi
done

#Deleting the DL files and N_samp files
rm gas_loading_prediction.h5 model.py training.csv
rm -rf 1* 2* 3* 4* 5* 6* 7* 8* 9*

# Move the unzip files into the WORKDIR
cd ..
/usr/bin/mv ${TMPDIR}/* ${WORKDIR}

# Removing __Pycache__
rm -rf ${WORKDIR}/__pycache__

# Delete tmp directory
cd ..
/usr/bin/rm -rf ${TMPDIR}
