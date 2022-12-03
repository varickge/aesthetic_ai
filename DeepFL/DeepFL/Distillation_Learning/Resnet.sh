#!/bin/bash
filename="log.txt"
if [ -f $filename ]; then 
   rm -i "$filename" 
fi

for epoch in {1..50}
    do
        python3 ResNet_Training.py -e $epoch
    done
        echo All done