#! /bin/bash
name=$1
if [ -z $name ]; then
    name="defualt"
fi

sudo /usr/local/cuda/bin/ncu --set full -f -o ./$name.ncu-rep main
