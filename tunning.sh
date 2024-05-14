#! /bin/bash

if [ ! -d ./tunning_res ]; then
    mkdir tunning_res
fi

log_file=tunning_res/res.log
touch $log_file
echo "" >$log_file

for k1 in 1 2 4 8 16; do
    for k2 in 1 2 4 8 16; do
        arg0=$(($k1 * 1048576))
        arg1=$(($k2 * 262144))
        echo "${arg0} ${arg1}" >>$log_file
        for i in 32 64 128 256 512 1024 2048; do
            sudo /usr/local/cuda/bin/ncu --set full -f -o tunning_res/hfuse$i.ncu-rep tunning.out $arg0 $arg1 $i
            mem=$(/usr/local/cuda/bin/ncu --import tunning_res/hfuse$i.ncu-rep | grep "Memory \[%\]" | grep -Eow "[0-9\.]+" | tr '\n' ',' | sed 's/,$//')
            com=$(/usr/local/cuda/bin/ncu --import tunning_res/hfuse$i.ncu-rep | grep "Compute (SM) \[%\]" | grep -Eow "[0-9\.]+" | tr '\n' ',' | sed 's/,$//')
            echo "gridDim[${i}] memory: ${mem} computation: ${com}" >>$log_file
            sudo rm tunning_res/hfuse$i.ncu-rep
        done
    done
done
