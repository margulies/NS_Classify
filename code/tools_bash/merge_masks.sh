#!/bin/bash

### In a directory with many masks, merges them and sets them each to a different intensity level

start=true
for mask in $1/*.nii.gz; do
	if [ "$start" = true ] ; then
		cp $mask $1/merged.nii.gz
		start=false
		i=2
	else
		fslmaths $mask -mul $i tmp
		fslmaths tmp -add $1/merged $1/output
		mv $1/output.nii.gz $1/merged.nii.gz
		i=`expr $i + 1`

	fi
done

rm tmp.nii.gz

