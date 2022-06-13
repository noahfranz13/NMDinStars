# imports                                                                                          
import os, sys, glob
import pandas as pd
import numpy as np

from mesa_reader import MesaData
from MesaOutput import MesaOutput

def convert(filepath):
    '''
    get metallicity from m_H and m_FE
    '''

    mesa = MesaData(filepath)
    
    # mass fractions
    mH = mesa.surface_h1[-1] # at the TRGB (last stage in evolution)
    mFE = mesa.surface_fe56[-1]
    
    # do the math
    # CONSTANTS
    # These are from wikipedia, cite them later!!
    molarMassFE = 55.845
    molarMassH = 1.00784
    
    # from: gs98
    solarFEH = -4.5
    
    feh = np.log10((mFE/mH)*(molarMassH/molarMassFE)) - solarFEH 
    
    return mesa.log_g[-1], mesa.effective_T[-1], feh, mesa.log_L[-1]


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="mesa output file", default='/media/ubuntu/T7/mesa-1296')
    args = parser.parse_args()

    m = MesaOutput(args.dir)
    badData = m.onlyConverging()
    
    outDict = {'flag':np.zeros(len(dataFiles), dtype=int), 'log_g': [],
               'Teff': [], '[Fe/H]': [], 'log_L': []}
    for f in m.dataPaths:
        logg, Teff, feh, logL =  convert(f)
        outDict['log_g'].append(logg)
        outDict['Teff'].append(Teff)
        outDict['[Fe/H]'].append(feh)
        outDict['log_L'].append(logL)
    #print(outDict)
    df = pd.DataFrame(outDict)
    df.to_csv("WorthyLeeBC/iBandOutput.txt", header=False,
              index=False, sep='\t')

if __name__ == '__main__':
    sys.exit(main())
