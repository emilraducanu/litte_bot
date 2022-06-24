#!/bin/bash
CMD="python3 ../parseur_moliere_new.py"
#loop to convert multiple files 
for  file  in  *.txt; do
     $CMD   "$file"
done
cat *_res.txt > corps.csv
exit 0
