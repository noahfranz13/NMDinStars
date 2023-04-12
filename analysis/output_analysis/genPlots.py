'''
Script to generate some interesting plots of the grid outputs
'''
# imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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

#sb.set(context='paper', style='whitegrid', palette='Set1')
'''
The rest of this is my own plotting code
'''
def io(mesaOutFile):

    sb.set(style='white', context='talk', palette='Set1') # Dark2
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.grid.which"] = 'both'
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'

    df = pd.read_csv(mesaOutFile, index_col=0)
    return df

def plotMI(mags):
    '''
    Plot mu_12 vs. I-band magnitude

    mags [DataFrame] : pandas dataframe of output file
    '''
    
    magsGood = mags[mags['flag'] == 0]

    a = 0.25
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    group = magsGood.groupby(magsGood.mu).mean().reset_index()
    std = magsGood.groupby(magsGood.mu).std()

    x = group.mu.to_numpy()
    y = group.M_I.to_numpy()
    err = std.M_I.to_numpy()
    
    ax.plot(x, y, '-', label=r'Average M$_I$')
    ax.fill_between(x, y-err, y+err, label=r'1$\sigma$ M$_I$', alpha=0.25)
    #ax.errorbar(magsGood.mu, magsGood.M_I, yerr=magsGood.M_I_err, fmt='.', label=r'M$_I$')

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax+0.5)
    
    #ax.plot(x, -3.96*np.ones(len(x)), linestyle='--', color='k', label=r'$\omega$ Centauri')
    #ax.fill_between(x, -3.96-0.05, -3.96+0.05, color='k', alpha=a)
    #ax.plot(x, -4.027*np.ones(len(x)), linestyle='--', color='orange', label=r'NGC4258')
    #ax.fill_between(x, -4.027-0.055, -4.027+0.055, color='orange', alpha=a)
    #ax.plot(x, -4.047*np.ones(len(x)), linestyle=':', color='royalblue', label=r'LMC (F20)')
    #ax.fill_between(x, -4.047-0.045, -4.047+0.045, color='royalblue', alpha=a)
    ax.plot(x, -3.958*np.ones(len(x)), linestyle=':', color='k', label=r'LMC (Y19)')
    ax.fill_between(x, -3.958-0.046, -3.958+0.046, color='k', alpha=a)

    
    ax.set_xlabel(r'$\mu_{12}$')
    ax.set_ylabel('I-Band Magnitude')
    #ax.set_xscale('log')
    ax.set_xlim(xmin, xmax+0.5)
    ax.legend(prop={'size': 10}, loc='best')
    fig.savefig('mu12_vs_MI.jpeg', transparent=False,
                bbox_inches='tight')

def histFlags(mags):
    '''
    Histogram the flags outputted by post process

    mags [DataFrame] : pandas dataframe of output file
    '''
    fig, ax = plt.subplots()
    ax.hist(mags.flag)
    ax.set_xlabel('Flag')
    ax.set_ylabel('N')

    fig.savefig('flagHist.jpeg', transparent=False,
                bbox_inches='tight')

def histAll(mags):
    '''
    Create histograms of I-Band and I-Band Error
    '''
    mags = mags[mags.flag==0]
    labels = ['I-Band Magnitude', 'I-Band Magnitude Error',
              'Mass', 'Y', 'Z', r'$\mu_{12}$']
    keys = ['M_I', 'M_I_err', 'mass', 'y', 'z', 'mu']
    for key, label in zip(keys, labels):
        fig, ax = plt.subplots(1, figsize=(8,6))
        ax.hist(mags[key])
        ax.set_ylabel('Number of Models')
        ax.set_xlabel(label)
        ax.grid()
        fig.savefig(key+'_hist.jpeg', transparent=False, bbox_inches='tight')

def plot4d(mags):
    '''
    Plot m, y, z, and mu all together
    '''
    import cmasher as cmr
    
    magsGood = mags[mags['flag'] == 0]

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    #fig, ax = plt.subplots(figsize=(8,6))
    img = ax.scatter(magsGood.mass,
                     magsGood.y,
                     magsGood.z,
                     c=magsGood.M_I,
                     s=magsGood.mu,
                     cmap=cmr.dusk)
    fig.colorbar(img, label=r'$\mu_{12}$')
    ax.set_xlabel('Mass')
    ax.set_ylabel('Helium Fraction')
    ax.set_zlabel('Metallicity')
    
    fig.savefig('allParams.jpeg', transparent=False,
                bbox_inches='tight')

def Iband_vs_binned(df, obs, useAllMus=True, restrictY=False):
    '''
    Plots M_I vs. binned version of other input params
    '''
    df = df[df.flag==0]

    labels = [r'Mass [M$_\odot$]', 'Y', 'Z']
    keys = ['mass', 'y', 'z']
    tols = [0, 0, 0]
    locs = ['upper left', 'upper left', 'best']
    allMus = np.sort(df.mu.unique())

    
    if useAllMus:
        mus = [allMus[1], allMus[-10], allMus[-1]]
    else:
        mus = [allMus[-1]]

    if obs == 'NGC4258':
        obsI = [-4.027]
        obsErr = [0.055]
        obsNew = [obs]
    elif obs == 'LMC_F20':
        obsI = [-4.047]
        obsErr = [0.045]
        obsNew = [obs]
    elif obs == 'LMC_Y19':
        obsI = [-3.958]
        obsErr = [0.046]
        obsNew = [obs]
    elif obs == 'OmegaCentauri':
        obsI = [-3.96]
        obsErr = [0.05]
        obsNew = [obs]
    elif obs == 'uncorrected':
        obsI = [-4.027, -4.047, -3.958, -3.96]
        obsErr = [0.055, 0.045, 0.046, 0.05]
        obsNew = ['NGC4258', 'LMC F20', 'LMC Y19', 'Omega Centauri']
    else:
        raise ValueError('Please enter a valid observational calibration: NGC4258, LMC_F20, LMC_Y19, or OmegaCentauri')
            
    Iband = df.M_I.to_numpy()
    VI = df.V_I.to_numpy()
    Ierr = df.M_I_err.to_numpy()
    VIerr = df.V_I_err.to_numpy()
    cov_I_VI = np.cov(Ierr, VIerr)[1,0]
    # NGC4258_Correction(denormIBand, denormIErr, denormVIBand, denormVIErr, yerr, cov_I_VI)
    if obs == 'NGC4258':
        corrected_IBand, sigma_2 = NGC4258_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'LMC_F20':
        corrected_IBand, sigma_2 = F20_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'LMC_Y19':
        corrected_IBand, sigma_2 = Y19_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'OmegaCentauri':
        corrected_IBand, sigma_2 = wCen_Correction(Iband, Ierr, VI, VIerr, obsErr, cov_I_VI)
    elif obs == 'uncorrected':
        corrected_IBand = Iband
        sigma_2 = None
    else:
        raise ValueError('Please enter a valid observational calibration: NGC4258, LMC_F20, LMC_Y19, or OmegaCentauri')

    df['M_I_corrected'] = pd.Series(corrected_IBand, index=df.index)
    
    for key, label, tol, loc in zip(keys, labels, tols, locs):

        fig, ax = plt.subplots(1, figsize=(8,6))
    
        for mu in mus:
            mags = df[df.mu == mu]
            #print(mags)

            group = mags.groupby([key]).mean().reset_index()
            std = mags.groupby([key]).std().reset_index()
            
            if mu == allMus[0]:
                cap = 'SM'
            else:
                cap = r'$\mu_{12}=$'+str(round(mu, 3))
            
            x = group[key].to_numpy()
            y = group.M_I_corrected.to_numpy()
            err = std.M_I_corrected.to_numpy()
            #print(x,y)
            ax.plot(x, y, '-', label=cap)
            ax.fill_between(x, y-err, y+err, label=r'1$\sigma$ {}'.format(cap), alpha=0.5)
            if not useAllMus:
                ax.fill_between(x, y-2*err, y+2*err, label=r'2$\sigma$ {}'.format(cap), alpha=0.25)
                ax.fill_between(x, y-3*err, y+3*err, label=r'3$\sigma$ {}'.format(cap), alpha=0.25)
                ax.fill_between(x, y-4*err, y+4*err, label=r'4$\sigma$ {}'.format(cap), alpha=0.25)
                ax.fill_between(x, y-5*err, y+5*err, label=r'5$\sigma$ {}'.format(cap), alpha=0.25)
            
                
        # Plot observational values
        a = 0.25
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax+tol)

        colors = ['k', 'orange', 'yellow', 'pink']
        for I, err, L, c in zip(obsI, obsErr, obsNew, colors):
            ax.plot(x, I*np.ones(len(x)), linestyle=':', color=c, label=L)
            ax.fill_between(x, I-err, I+err, color=c, alpha=a)
        
        ax.set_xlabel(label)
        ax.set_ylabel('I-Band Magnitude')
        ax.legend(prop={'size': 10}, loc=loc, ncol=2)
        ax.set_xlim(xmin, xmax+tol)
        if restrictY:
            ax.set_ylim(-4.5, -3.8)
        
        if useAllMus:
            figpath = f'M_I_vs_binned_{key}.jpeg'
        else:
            figpath = f'M_I_vs_binned_{key}_mu6.jpeg'
        fig.savefig(figpath, transparent=False,
                    bbox_inches='tight')
    
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help='file to plot stuff from')
    parser.add_argument('--obs', help='observational correction to use')
    parser.add_argument('--restrictY', dest='restrictY', action='store_true', help='should we restrict the y limits')
    parser.set_defaults(restrictY=False)
    args = parser.parse_args()
    
    df = io(args.infile)
    plotMI(df)
    histFlags(df)
    plot4d(df)
    histAll(df)
    Iband_vs_binned(df, args.obs, restrictY=args.restrictY)
    Iband_vs_binned(df, args.obs, useAllMus=False, restrictY=args.restrictY)
    
if __name__ == '__main__':
    sys.exit(main())
