'''
Script to align Mitchell Dennis's SM grid with my own to get values
with mu_12=0
'''
# imports
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ubuntu/Documents/NMDinStars/ML_models/')
from MD_machineLearningFunctions import deNormalise

# first read in both files
myOutFile1 = 'postProcess_output_first.txt'
myOutFile2 = 'postProcess_output_second.txt'
mtOutFile3 = 'postProcess_output_third.txt'
myOutDir = '/home/ubuntu/Documents/NMDinStars/mesa_mod/mesa-r12778/star/test_suite/NeutrinoMagneticDipoleMesa/WorthyLeeBC/'
myOutPath1 = os.path.join(myOutDir, myOutFile1)
myOutPath2 = os.path.join(myOutDir, myOutFile2)
myOutPath3 = os.path.join(myOutDir, myOutFile3)
smOutPath = '/home/ubuntu/Documents/NMDinStars/ML_models/fulldata.txt' 

out1 = pd.read_csv(myOutPath1, index_col=0)
out2 = pd.read_csv(myOutPath2, index_col=0)
out3 = pd.read_csv(myOutPath3, index_col=0)
out = pd.concat([out1, out2, out3])
sm = pd.read_csv(smOutPath)

# denormalize using MD's denormalization algorithm
keyOrder = ['mass', 'Y', 'Z', 'time', 'grav', 'Teff', 'FeH', 'logL', 'IBand', 'Ierr']
constPath = '/home/ubuntu/Documents/NMDinStars/ML_models/normalisationConstants.txt'
with open(constPath, 'r') as f:
    for line, key in zip(f, keyOrder):
        absMin, minVal, maxVal = line.split('\t')
        absMin = float(absMin)
        minVal = float(minVal)
        maxVal = float(maxVal)

        sm[key] = deNormalise(minVal, maxVal, absMin, sm[key])
        
# Get unique m,y,z
m = out.mass.unique()
y = out.y.unique()
z = out.z.unique()

# loop over each of these to find row in SM grid with minimum
# difference from them

print(out)
print(sm)

rowsToAdd = []
for mm in m:
    for yy in y:
        for zz in z:
            mDiff = np.array(abs(mm - sm.mass))
            yDiff = np.array(abs(yy - sm.Y))
            zDiff = np.array(abs(zz - sm.Z))
            
            whereM = np.where(min(mDiff) == mDiff)[0]
            whereY = np.where(min(yDiff) == yDiff)[0]
            whereZ = np.where(min(zDiff) == zDiff)[0]

            mIdx = sm.iloc[whereM].i.unique()[0]
            yIdx = sm.iloc[whereY].j.unique()[0]
            zIdx = sm.iloc[whereZ].k.unique()[0]
            
            row = sm[(sm.i==mIdx) & (sm.j==yIdx) & (sm.k==zIdx)]

            if len(row) != 0:
                #print(mm, yy, zz)
                #print(mIdx, yIdx, zIdx)

                rowsToAdd.append(row)

outDf = pd.concat(rowsToAdd, axis=0)

# add new columns for mu and mu_idx
outDf['mu_index'] = np.ones(len(outDf))*-1
outDf['mu'] = np.zeros(len(outDf))

# align column names
outDf.drop('time', axis=1, inplace=True)
outDf.columns = [ 'flag', 'worthey_lee_flag', 'mass_index', 'y_index',
                  'z_index', 'mass', 'y', 'z', 'surface_grav', 'Teff',
                  'feh', 'L', 'M_I', 'M_I_err', 'mu_index', 'mu']

outDf['index'] = -1*np.arange(1, len(outDf)+1, 1)
outDf.set_index('index', inplace=True)

totGrid = pd.concat([out, outDf], axis=0)
print(totGrid)
totGrid.to_csv('allData.csv')
