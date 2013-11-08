#! /bin/bash

# Script separates masks that are in coded as activation levels

if [ ! $1 ]; then
    echo "Usage: $0 <input>"    
    echo "Output: n separated masks"
    exit 77
fi

### SETUP
WD=`mktemp -d -t parse` || exit 79

out=`fslstats $1 -R`

MAX=`awk '{print $2}' <(echo $out) | awk -F'.' '{print $1}'`

mkdir ${1//.nii.gz}

for i in $(seq 1 $MAX)
do
	fslmaths $1 -thr $i -uthr $i -bin ${1//.nii.gz}/$i

done