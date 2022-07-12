'''
Script to generate some interesting plots of the grid outputs
'''
# imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.set(context='talk', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

def io(mesaOutFile):
    return pd.read_csv(mesaOutFile, index_col=0)

def plotMI(mags):
    '''
    Plot mu_12 vs. I-band magnitude

    mags [DataFrame] : pandas dataframe of output file
    '''
    
    magsGood = mags[mags['flag'] == 0]

    x = np.linspace(0, max(magsGood.mu)+0.5)
    a = 0.25
    
    fig, ax = plt.subplots()
    group = magsGood.groupby(magsGood.mu).mean().reset_index()
    std = magsGood.groupby(magsGood.mu).std()
    
    ax.errorbar(group.mu, group.M_I, yerr=std.M_I, fmt='o', label=r'Average M$_I$', capsize=6)
    #ax.errorbar(magsGood.mu, magsGood.M_I, yerr=magsGood.M_I_err, fmt='.', label=r'M$_I$')

    #ax.plot(x, -3.96*np.ones(len(x)), linestyle='--', color='k', label=r'Cappozi & Raffelt $\omega$ Centauri')
    #ax.fill_between(x, -3.96-0.05, -3.96+0.05, color='k', alpha=a)
    #ax.plot(x, -4.027*np.ones(len(x)), linestyle='--', color='orange', label=r'Cappozi & Raffelt NGC4258')
    #ax.fill_between(x, -4.027-0.055, -4.027+0.055, color='orange', alpha=a)
    ax.plot(x, -4.047*np.ones(len(x)), linestyle=':', color='royalblue', label=r'Cappozi & Raffelt LMC (F20)')
    ax.fill_between(x, -4.047-0.045, -4.047+0.045, color='royalblue', alpha=a)
    #ax.plot(x, -3.958*np.ones(len(x)), linestyle=':', color='green', label=r'Cappozi & Raffelt LMC (Y19)')
    #ax.fill_between(x, -3.958-0.046, -3.958+0.046, color='green', alpha=a)

    ax.set_xlabel(r'$\mu_{12}$')
    ax.set_ylabel('I-Band Magnitude')
    ax.set_xscale('log')
    ax.legend(prop={'size': 10})
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

def Iband_vs_binned(mags):
    '''
    Plots M_I vs. binned version of other input params
    '''
    mags = mags[mags.flag==0]
    #mags = mags[mags.index > 0]
    
    labels = ['Mass', 'Y', 'Z']
    keys = ['mass', 'y', 'z']
    tols = [0.05, 0.01, 1e-5]
    
    for key, label, tol in zip(keys, labels, tols):

        group = []
        std = []
        for i in range(0, len(mags[key].unique()), 2):
            m = mags[key].unique()[i]
            
            where = np.where(abs(mags[key] - m) <= tol)[0]
            good = mags.iloc[where]
            group.append(np.mean(good))
            std.append(np.std(good))
            
        group = pd.concat(group, axis=1).T
        std = pd.concat(std, axis=1).T
        
        fig, ax = plt.subplots(1, figsize=(8,6))
        ax.errorbar(group[key], group.M_I, yerr=std.M_I, fmt='o', capsize=6)
        ax.set_xlabel(label)
        ax.set_ylabel('I-Band Magnitude')
        if key == 'z':
            ax.set_xscale('log')
        fig.savefig(f'M_I_vs_binned_{key}.jpeg', transparent=False,
                bbox_inches='tight')
    
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help='file to plot stuff from')
    args = parser.parse_args()

    df = io(args.infile)
    plotMI(df)
    histFlags(df)
    plot4d(df)
    histAll(df)
    Iband_vs_binned(df)
    
if __name__ == '__main__':
    sys.exit(main())
