import re
import regex
import os
import sys

# filename = "/home/user/litte-bot/new/corp/corneillep_menteur.txt"
filename = sys.argv[1]
# garder seulement le nom de fichier sans path.
title = os.path.basename(filename)
title = re.sub(".txt", "", title)
str = ""
person = ""  # la personne qui parle.
i = 0
fin = open(title + "_input.txt", "w")
fout = open(title + "_target.txt", "w")
fres = open(title + "_res.txt", "w")
p1 = []
p2 = []

with open(filename, 'r') as f:
    pr1 = []
    pr2 = []
    begin = False
    start = False
    input = False
    for line in f:
        if not start:
            if "SCÈNE" in line or "SCENE" in line:
                start = True
            continue
        if line == "\n" or "#" in line or "<" in line or "SCÈNE" in line or "SCENE" in line or "===" in line:
            continue
        if regex.match('^\w[[:upper:]].|,', line.strip()):  # ligne avec la personne
            input = not input
            person = line.strip()
            person = re.sub('\.', '', person) # supprimer les points
            person = re.sub(',.*', '', person) # supprimer la personne a qui parle
            # print(i, person)
            # print(person + "\n")
            if input:
                fin.write(person + "\n")
                p1.append(pr1)
                pr1 = []
            else:
                fout.write(person + "\n")
                p2.append(pr2)
                pr2 = []
            begin = True
            i = 0
            continue
        if begin == True:
            str = line.strip().lower()
            str = "".join(str)
            # print(str)
            i += 1
            if i % 2 == 0:
                str += "\n"
            if input:
                fin.write(str + '\n')
                pr1.append(str.strip())
            else:
                fout.write(str + '\n')
                pr2.append(str.strip())
                # print('\n')

fin.close()
fout.close()

np1 = []
np2 = []
for i in range(len(p1)):
    t1 = []
    for n in range(0, len(p1[i]), 2):
        if ((n + 1) % len(p1[i]) != 0):
            t1.append(p1[i][n] + ' ' + p1[i][n + 1])
        else:
            t1.append(p1[i][n])
    if len(t1) != 0:
        np1.append(t1)
        
for i in range(len(p2)):
    t2 = []
    for n in range(0, len(p2[i]), 2):
        if ((n + 1) % len(p2[i]) != 0):
            t2.append(p2[i][n] + ' ' + p2[i][n + 1])
        else:
            t2.append(p2[i][n])
    if len(t2) != 0:
        np2.append(t2)

# print(len(np1))
# print(len(np2))
m = min(len(np1), len(np2))
for i in range(m):
    maxi = max(len(np1[i]), len(np2[i]))
    for n in range(maxi):
        fres.write('"' + np1[i][n % len(np1[i])] + '"' + ',' + '"' + np2[i][n % len(np2[i])] + '"' + '\n')

fres.close()
