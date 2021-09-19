#!/bin/bash

for phi in 0 0.6; do
    for run in {1..10}; do
        python ./run.py $phi simple $run > /dev/null &
    done
done
