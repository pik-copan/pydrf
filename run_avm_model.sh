#!/bin/bash

cc=0
n_ops=2
tc=1
steps=89
phi=0.0

# for phi in 0.0 0.6; do
# for run in {1..10}; do
# 	python run_avm_model.py $phi $cc $n_ops $tc $steps $run &
# done      
# done

steps=255
phi=0.6

for run in {1..10}; do
    python run_avm_model.py $phi $cc $n_ops $tc $steps $run 
done      

#cc=1
#phi=0
#python ./run.py $phi $cc $n_ops $survey_size $tc $start_time >> "log/$phi _$cc _$n_ops _$survey_size.log" &
