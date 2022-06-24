import re
import os
import sys
import regex

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
pr1 = []
pr2 = []

def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x

with open(filename, 'r') as f:
    begin = False
    start = False
    skip = False
    input = False
    for line in f:
        if skip:
            skip = not skip
            continue
        if "Scène" in line:
            start = True
            skip = True
        if not start:
            continue
        if line == "\n" or "#" in line or "<" in line or "SCÈNE" in line or "===" in line:
            continue
        if re.match('    ', line):  # ligne avec la personne
            input = not input
            person = line.strip()
            person = re.sub('\.', '', person) # supprimer les points
            person = re.sub(',.*', '', person) # supprimer les personne a qui parle
            if input:
                fin.write(person + "\n")
            else:
                fout.write(person + "\n")
            begin = True
            i = 0
            continue
        if begin == True:
            str = line.strip().lower()
            str = "".join(str)
            str = re.sub('"', '', str)
            if len(str) > 120:
                str = re.sub(', ', '\n', str)
                str = re.sub('\. ', ".\n", str)
                str = re.sub('\.\* ', ".\n*", str)
                str = re.sub(': ', ':\n', str)
                str = re.sub(', \*', ',\n*', str)
                str = re.sub('\? ', '?\n', str)
                str = re.sub("! ", "!\n", str)
                str = re.sub('et ', '\net ', str)
                str = re.sub(', \*', ',\n*', str)
                str = re.sub('; ', '\n', str)
                str = re.sub('mais ', '\nmais ', str)
                splited = str.split('\n')
                str = ""
                strl = ""
                i = 0
                enter = False
                final = False
                while i < len(splited):
                    strl = splited[i].strip()
                    while len(strl) <= 60:
                        i += 1
                        if i >= len(splited):
                            final = True
                            break
                        strl += ' ' + splited[i]
                        enter = True
                    if final:
                        break
                    strl += '\n'
                    enter = False
                    i += 1
                    str += strl
                # print(str)
            i += 1
            # if i % 2 == 0:
            #     str += "\n"
            if input:
                fin.write(chomp(str) + '\n')
            else:
                fout.write(chomp(str) + '\n')
fin.close()
fout.close()

with open(title + "_input.txt", 'r') as f:
    for line in f:
        if regex.match('^\w[[:upper:]]', line):  # ligne avec la personne
            p1.append(pr1)
            pr1 = []
            i = 0
            continue
        pr1.append(chomp(line))

with open(title + "_target.txt", 'r') as f:
    for line in f:
        if regex.match('^\w[[:upper:]]', line):  # ligne avec la personne
            p2.append(pr2)
            pr2 = []
            i = 0
            continue
        pr2.append(chomp(line))


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
