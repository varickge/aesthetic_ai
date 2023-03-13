#!/bin/bash
filename="log_class.txt"
if [ -f $filename ]; then
    rm -i "$filename"
fi

# Basic range in for loop
for generation in {0..100}
    do 
        python GA_custom.py -g $generation
    done 
        echo ALL done