#!/bin/bash
source activate ~/master/
declare -a arr=('burgers' 'sports' 'grocery' 'vegies')
for loc in "${arr[@]}"; do
	python ./calc_positions.py $loc >> "$loc.log" &
done
