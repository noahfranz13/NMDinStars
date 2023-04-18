# plot up the testing data

from MesaOutput import MesaOutput

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob

'''
The following 4 functions are from MD 
They quantify the V-I band correction for each object

denormIBand : the denormalized ML prediciton of the IBand
denormIerr  : the denormalized ML prediction of the Ierr
denormVIBand: The denormalized ML predication of the V-I band
denormVIErr : The denormalized ML prediction on the V-I band error
yerr        : The error on the observed value for the I-band magnitude
'''
def F20_Correction(denormIBand, denormIErr, denormVIBand, denormVIErr, yerr, cov_I_VI):
    partial_V = 0
    partial_I = 1
    sigma_MI_2 = (partial_I**2)*(denormIErr**2) + (np.abs(partial_V**2))*(denormVIErr)**2 + 2*partial_I*np.abs(partial_V)*denormVIErr*denormIErr*cov_I_VI
    sigma_2 = (yerr**2 + sigma_MI_2**2)
    corrected_IBand = denormIBand + 0.00*(denormVIBand - 1.8)
    return corrected_IBand, sigma_2

def Y19_Correction(denormIBand, denormIErr, denormVIBand, denormVIErr, yerr, cov_I_VI):
    partial_V = -0.182*(denormVIBand)-0.266
    partial_I = 1
    sigma_MI_2 = (partial_I**2)*(denormIErr**2) + (np.abs(partial_V**2))*(denormVIErr)**2 + 2*partial_I*np.abs(partial_V)*denormVIErr*denormIErr*cov_I_VI
    sigma_2 = (yerr**2 + sigma_MI_2**2)
    corrected_IBand = denormIBand - 0.091*(denormVIBand - 1.5)**2 + 0.007*(denormVIBand - 1.5)
    return corrected_IBand, sigma_2

def NGC4258_Correction(denormIBand, denormIErr, denormVIBand, denormVIErr, yerr, cov_I_VI):
    partial_V = -0.182*(denormVIBand)-0.266
    partial_I = 1
    sigma_MI_2 = (partial_I**2)*(denormIErr**2) + (np.abs(partial_V**2))*(denormVIErr)**2 + 2*partial_I*np.abs(partial_V)*denormVIErr*denormIErr*cov_I_VI
    sigma_2 = (yerr**2 + sigma_MI_2**2)
    corrected_IBand = denormIBand - 0.091*(denormVIBand - 1.5)**2 + 0.007*(denormVIBand - 1.5)
    return corrected_IBand, sigma_2

def wCen_Correction(denormIBand, denormIErr, denormVIBand, denormVIErr, yerr, cov_I_VI):
    partial_V = 0.16*denormVIBand - 0.046
    partial_I = 1
    sigma_MI_2 = np.abs(partial_I**2)*(denormIErr**2) + (np.abs(partial_V**2))*(denormVIErr)**2 + 2*np.abs(partial_I)*np.abs(partial_V)*denormVIErr*denormIErr*cov_I_VI
    sigma_2 = (yerr**2 + sigma_MI_2**2)
    corrected_IBand = denormIBand - 0.046*(denormVIBand-1.5) - 0.08*(denormVIBand-1.5)**2
    return corrected_IBand, sigma_2

def printIband(df, obs='NGC4258'):
    '''
    print out Iband info
    '''

    Iband = df['M_I']
    Ierr = df.M_I_err
    VI = df.V_I
    VIerr = df.V_I_err
    cov_I_VI = np.cov(Ierr, VIerr)[1,0]
    
    if obs == 'NGC4258':
        obsI = -4.027
        obsErr = 0.055
    elif obs == 'LMC_F20':
        obsI = -4.047
        obsErr = 0.045
    elif obs == 'LMC_Y19':
        obsI = -3.958
        obsErr = 0.046
    elif obs == 'OmegaCentauri':
        obsI = -3.96
        obsErr = 0.05
    else:
        raise ValueError('Please enter a valid observational calibration: NGC4258, LMC_F20, LMC_Y19, or OmegaCentauri')
    
    if obs == 'NGC4258':
        corrected_IBand, sigma_2 = NGC4258_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'LMC_F20':
        corrected_IBand, sigma_2 = F20_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'LMC_Y19':
        corrected_IBand, sigma_2 = Y19_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'OmegaCentauri':
        corrected_IBand, sigma_2 = wCen_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    else:
        raise ValueError('Please enter a valid observational calibration: NGC4258, LMC_F20, LMC_Y19, or OmegaCentauri')

    df['M_I_corrected'] = pd.Series(corrected_IBand, index=df.index)

    filepaths = df.filepath
    df.drop(columns='filepath', inplace=True)

    # /home/nfranz/lus_scratch/mesa-testing/MassOf1.2/nmdm_M1.2_Y0.25_Z0.02_U6.data

    Ms, Ys, Zs, Us = [], [], [], []
    for f in filepaths:
        filename = os.path.split(f)[-1]
        Ms.append(float(filename.split('_')[1][1:]))
        Ys.append(float(filename.split('_')[2][1:]))
        Zs.append(float(filename.split('_')[3][1:]))
        Us.append(float(filename.split('_')[4].split('.')[0][1:]))

    df['M'] = Ms
    df['Y'] = Ys
    df['Z'] = Zs
    df['mu12'] = Us

    cols = ['M', 'Y', 'Z', 'mu12', 'M_I', 'M_I_corrected']
    print(df[cols])
    return df[cols]
    
def plot(m, ax, idx=0, **kwargs):
    '''
    '''
    ax.plot(m.effective_T[idx:], 10**m.log_L[idx:], **kwargs)
    ax.set_xlabel('Effective Temperature [K]')
    ax.set_ylabel(r'Luminosity (L$_\odot$)')
    ax.set_yscale('log')
    ax.invert_xaxis()
    #ax.set_ylim(3, max(10**m.log_L)+10**3)

def main():

    #dirpath = '/home/nfranz/research/data/mesa-testing'
    dirpath = '/home/nfranz/lus_scratch/mesa-testing/'

    mesa = MesaOutput(dirpath)

    MtoFind = 2
    masses = np.array([float(os.path.split(f)[-1].split('_')[1][1:]) for f in mesa.dataPaths])
    whereM = np.where(masses == MtoFind)[0]
    mesa = mesa[whereM]
                      
    mesa.data = [mesa.data[1], mesa.data[2], mesa.data[3], mesa.data[0]]
    
    fig, ax = plt.subplots(figsize=(16,12))
    for m in mesa:

        Ls = os.path.split(m.file_name)[-1].split('_')
        L = 'Z=' + Ls[3][1:] + r', $\mu_{12}$=' + Ls[4][1:].split('.')[0]
        
        plot(m, ax, idx=90, label=L, linewidth=7, alpha=0.8)

    ax.legend()
    fig.savefig('testing-plot.png', transparent=False, bbox_inches='tight')

    # print info on outputs
    df = pd.read_csv('/home/nfranz/NMDinStars/mesa_mod/mesa-r12778/star/test_suite/NeutrinoMagneticDipoleMesa/WorthyLeeBC/postProcess_output_mesa-testing.txt')
    newDf = printIband(df)
    newDf.to_csv('mesa-testing-info.csv')

if __name__ == '__main__':
    main()
