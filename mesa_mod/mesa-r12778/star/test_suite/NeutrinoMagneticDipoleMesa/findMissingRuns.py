# imports
import os, glob
import numpy as np
import pandas as pd

# collect grid
gridFiles = glob.glob('/home/nfranz/NMDinStars/makeGrids/third*.txt')
grids = []
for g in gridFiles:
    grids.append(pd.read_csv(g, header=None, sep='\t', index_col=0))

grid = pd.concat(grids)

dirs = glob.glob('/home/nfranz/lus_scratch/mesa-112500/third/out*/*.data')

idxList = np.array([int(os.path.split(d)[0].split('/')[-1].split('i')[-1]) for d in dirs])
print(grid)
unfinishedGrid = [pd.DataFrame()]
for i in range(len(grid)):
    row = grid.iloc[i]
    #print(type(row))
    whereIdx = np.where(row.name == idxList)[0]
    
    if len(whereIdx) < 1:
        unfinishedGrid.append(row)

toRerun = pd.concat(unfinishedGrid, axis=1).T
print(toRerun.keys())
toRerun = toRerun.astype({1: int, 2: int, 3: int, 4: int})

print(toRerun.head())
toRerun.to_csv('grid_to_rerun.csv', header=None, sep='\t', float_format='%.15f')
