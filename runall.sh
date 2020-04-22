#!/bin/bash

# executes all jupyter notebooks to perform full data analysis from raw data to final figures

echo
echo "Warning: This will take quite some time!"
read -p "Are you sure you want to continue? [Y/N] " answer
echo
if [ $answer == "N" ]
then
    echo "Aborting"
elif [ $answer != "Y" ]
then
    echo "Invalid answer. Aborting"
else
    echo "Running data analysis"
    echo
    read -p "What should be the timeout in second? " maxtime 
    echo "Waiting for 10s. Last chance to abort..."
    sleep 10
    
    echo "Executing DC bias cavities files"
    rm -f model_DC_bias_cavity.md
    jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=$maxtime -y model_DC_bias_cavity.ipynb
    jupytext --to markdown model_DC_bias_cavity.ipynb
    
    #echo "Figures"
    #bash compilefigs.sh
    
    #echo "Documentation"
    #python python_info.py
    #bash tree_info.sh

fi
