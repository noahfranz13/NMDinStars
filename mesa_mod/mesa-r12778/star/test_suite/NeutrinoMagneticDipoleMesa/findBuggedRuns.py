# file to find .data files that are bugged and need to be rerun

from MesaOutput import MesaOutput

dirpath = '/home/nfranz/lus_scratch/sm/'
m = MesaOutput(dirpath)

corrupted = []
for ii, d in enumerate(m.data):
    try:
        max(d.star_age)
    except:
        corrupted.append(m.dataPaths[ii])

print(corrupted)
