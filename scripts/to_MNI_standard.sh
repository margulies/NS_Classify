#! /bin/bash

# Script converts to MNI standard bash

if [ ! $1 ]; then
    echo "Usage: $0 <input> <output>"    
    echo "Output: Converts Yeo input to MNI Standard"
    exit 77
fi

### SETUP
WD=`mktemp -d -t parse` || exit 79

input=$1
output=$2

fslreorient2std $1 $WD/temp.nii.gz

flirt -ref $WD/temp.nii.gz -in $WD/temp.nii.gz -out $WD/temp2.nii.gz -applyisoxfm 2

fslroi $WD/temp2 $output 18.5 91 9.5 109 18.5 91