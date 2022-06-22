import os, glob
import pandas as pd
import numpy as np
dirs = glob.glob('/home/nfranz/lus_scratch/mesa-1296/*')
gridFiles = glob.glob('/home/nfranz/NMDinStars/makeGrids/*.txt')
grids = []
for g in gridFiles:
    grids.append(pd.read_csv(g, header=None, sep='\t', index_col=0))

grid = pd.concat(grids)
print(grid)
for d in dirs:

    split = d.split('/')[-1].split('_')
    u = int(split[-2][1:])
    z = int(split[-3][1:])
    y = int(split[-4][1:])
    m = int(split[-5][1:])

    

    line = grid[(grid.iloc[:,0].astype(int)==m) & (grid.iloc[:,1].astype(int)==y) & (grid.iloc[:,2].astype(int)==z) & (grid.iloc[:,3].astype(int)==u)]
    idx = line.index[0]

    dNew = d.split('i')[0] + 'i' + str(idx)

    #os.rename(d, dNew)
    files = glob.glob(dNew + '/*')

    for f in files:

        path, fName = os.path.split(f)
        i = fName.find('_i')
        ext = fName.split('.')[-1]
        #print(fName[:i])
        fNew = path + '/' + fName[:i] + '_' + str(idx) + '.' + ext
        os.rename(f, fNew)
 
'''
for i, f in enumerate(datafiles):

    fOld = f
    ext = f.split('.')[-1]
    f = f.split('.')[0]
    fNew = f[:-7] + 'M' + f[-7:]
    fNew = fNew[:-5] + 'Y' + fNew[-5:]
    fNew = fNew[:-3] + 'Z' + fNew[-3:]
    fNew = fNew[:-1] + 'U' + fNew[-1]
    fNew += f'_i{i}'
    fNew += '.' + ext
    idx = f.split('/')[-2].split('i')[-1]
    fSplit = f.split('/')
    fNew = fSplit[-1].split('i')[0] + 'i' + idx + '.' + ext
    f1 = "/"
    for item in fSplit[:-1]:
        f1 = os.path.join(f1, item)
    fNew = os.path.join(f1, fNew)
    print(fOld, fNew)

    os.rename(fOld, fNew)

for i, f in enumerate(textfiles):

    fOld = f
    ext = f.split('.')[-1]
    f = f.split('.')[0]
    fNew = f[:-7] + 'M' + f[-7:]
    fNew = fNew[:-5] + 'Y' + fNew[-5:]
    fNew = fNew[:-3] + 'Z' + fNew[-3:]
    fNew = fNew[:-1] + 'U' + fNew[-1]
    fNew += f'_i{i}'
    fNew += '.' + ext
    
    idx = f.split('/')[-2].split('i')[-1]
    fSplit = f.split('/')
    fNew = fSplit[-1].split('i')[0] + 'i' + fSplit[-1].split('i')[1] + 'i' + idx + '.' + ext
    #print(fNew)
    f1 = "/"
    for item in fSplit[:-1]:
        f1 = os.path.join(f1, item)
    fNew = os.path.join(f1, fNew)
    print(fOld, fNew)

    os.rename(fOld, fNew)
'''

