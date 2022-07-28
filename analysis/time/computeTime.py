# imports
file = '/media/ubuntu/T7/NF/mesa-112500/runtimes.txt'

tot = 0
with open(file, 'r') as f:
    for line in f:
        if len(line) > 1:
            tot += float(line)

print(tot/(60*24*365), ' yrs')
