import re
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
pr1 = []
pr2 = []

with open(filename, 'r') as f:
    begin = False
    start = False
    input = False
    for line in f:
        if not start:
            if "SCÈNE" in line:
                start = True
            continue
        if line == "\n" or "#" in line or "<" in line or "SCÈNE" in line or "===" in line:
            continue
        if re.match("^\w[A-Z]", line):  # ligne avec la personne
            input = not input
            person = line.strip()
            person = re.sub('\.', '', person) # supprimer les points
            person = re.sub(',.*', '', person) # supprimer les personne a qui parle
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
            # if i % 2 == 0:
                # str += "\n"
            if input:
                fin.write(str + '\n')
                pr1.append(str.strip())
            else:
                fout.write(str + '\n')
                pr2.append(str.strip())
                # print('\n')

fin.close()
fout.close()

print(len(p1))
print(len(p2))

for i in range(len(p1)):
    maxi = max(len(p1[i]), len(p2[i]))
    for n in range(maxi):
        fres.write('"' + p1[i][n % len(p1[i])] + '"' + ',' + '"' + p2[i][n % len(p2[i])] + '"' + '\n')
fres.close()
