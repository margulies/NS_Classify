#! /bin/bash

# Script separates masks that are in coded as activation levels

if [ ! $1 ]; then
    echo "Usage: $0 <input> <outputdir>"    
    echo "Output: n separated masks"
    exit 77
fi

MAX = `awk '{print $2}' <(echo $a)`

echo ${MAX:0}