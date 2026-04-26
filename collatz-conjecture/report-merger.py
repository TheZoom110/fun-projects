import csv
import re

with open('collatz-10pow6-finer-report.txt', 'r') as f:
    fread = f.read()

with open('collatz-10pow9-report.txt', 'r') as g:
    gread = g.read()

"""
Extracts (N, density) from terminal output lines like:
    N=  10,000,000  |  same-T: 4,964,705  | ...
Works regardless of spacing or comma formatting.
"""

fpairs = []
gpairs = []
pattern = r'N=\s*([\d,]+)\s*\|\s*same-T:\s*([\d,]+)'

for line in fread.splitlines():
    m = re.search(pattern, line)
    if m:
        n       = int(m.group(1).replace(',', ''))
        same_t  = int(m.group(2).replace(',', ''))
        fpairs.append((n, same_t))

for line in gread.splitlines():
    m = re.search(pattern, line)
    if m:
        n       = int(m.group(1).replace(',', ''))
        same_t  = int(m.group(2).replace(',', ''))
        gpairs.append((n, same_t))

i, j = 0, 0
mpairs = []

while i < len(fpairs) and j < len(gpairs):
    if fpairs[i][0] < gpairs[j][0]:
        mpairs.append(fpairs[i])
        i += 1
    elif fpairs[i][0] > gpairs[j][0]:
        mpairs.append(gpairs[j])
        j += 1
    else:
        mpairs.append(fpairs[i])
        i += 1
        j += 1

mpairs.extend(fpairs[i:])
mpairs.extend(gpairs[j:])

mpairs.insert(0, ("N", "Same-T"))

with open("collatz-10pow9-finer-report-merged.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(mpairs)