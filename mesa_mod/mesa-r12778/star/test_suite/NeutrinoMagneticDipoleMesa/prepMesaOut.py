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

def runChecks(m, checkTime):
    # 1) check for time
    if checkTime:
        print('Checking for files that ran out of time...')
        finished = m.checkTime()

        if not finished:
            print('Not all models have finished!')
            print('Exiting, please rerun necessary grid')
            sys.exit()
        
    # 2) flag ones with incorrect termination code
    print('Checking for files that did not converge...')
    m.checkConverging()

    # 3) Check for shell flash
    print('Checking for early shell flashing...')
    try:
        m.checkFlash()
    except:
        print('No center_he4 outputted, using modified method')
        m.MD_findShellFlash()

    # 4) Check for age cuts
    print('Checking the age of outputs...')
    m.checkAge()
    
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="mesa output file", default='/media/ubuntu/T7/mesa-1296')
    parser.add_argument('--noTimeCheck', dest='timeCheck', action='store_false')
    parser.add_argument('--noIndexSet', dest='idxSet', action='store_true')
    parser.set_defaults(idxSet=False)
    parser.set_defaults(useNMDM=True)
    args = parser.parse_args()

    m = MesaOutput(args.dir)
    
    # check the output data first!!!
    runChecks(m, args.timeCheck)
    print(m.flags)
    # Now write the output files
    outDict = {'flag':m.flags.astype(int), 'log_g': [],
               'Teff': [], '[Fe/H]': [], 'log_L': [], 'filepath':[]}
    for f in m.dataPaths:
        logg, Teff, feh, logL =  convert(f)
        outDict['log_g'].append(logg)
        outDict['Teff'].append(Teff)
        outDict['[Fe/H]'].append(feh)
        outDict['log_L'].append(logL)
        outDict['filepath'].append(f)
    
    df = pd.DataFrame(outDict)
    if not args.idxSet:
        df.set_index(m.index, inplace=True)
    df.to_csv("WorthyLeeBC/iBandOutput.txt", header=False,
              index=True, sep='\t')

if __name__ == '__main__':
    sys.exit(main())
