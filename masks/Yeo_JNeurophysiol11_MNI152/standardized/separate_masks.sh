#! /bin/bash

# Script separates masks that are in coded as activation levels

if [ ! $1 ]; then
    echo "Usage: $0 <input>"    
    echo "Output: n separated masks"
    exit 77
fi

MAX=`awk '{print $2}' <(fslstats 7Networks_Liberal.nii.gz -R)`
MAX=${MAX:0:1}

### SETUP
WD=`mktemp -d -t parse` || exit 79

for i in $(seq 1 $MAX); do
	fslmaths $1 -thr $i -uthr $i $WD/tmp
	fslmaths $WD/tmp -bin ${1%.nii.gz}_${i}
done
