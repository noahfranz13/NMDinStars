'''
Script to generate some interesting plots of the grid outputs
'''
# imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#sb.set(context='paper', style='whitegrid', palette='Set1')

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
    
    ax.errorbar(group.mu, group.M_I, yerr=std.M_I, fmt='o', label=r'Average M$_I$', capsize=4)
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
    ax.legend(prop={'size': 14})
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

def Iband_vs_binned(df):
    '''
    Plots M_I vs. binned version of other input params
    '''
    df = df[df.flag==0]
    
    labels = [r'Mass [M$_\odot$]', 'Y', 'Z']
    keys = ['mass', 'y', 'z']
    tols = [0, 0, 0.01]
    allMus = np.sort(df.mu.unique())
    mus = [allMus[1], allMus[-10], allMus[-1]]

    for key, label, tol in zip(keys, labels, tols):

        fig, ax = plt.subplots(1, figsize=(8,6))
    
        for mu in mus:
            mags = df[df.mu == mu]
            #print(mags)

            group = mags.groupby([key]).mean().reset_index()
            std = mags.groupby([key]).std().reset_index()

            if mu == mus[0]:
                cap = 'SM'
            else:
                cap = r'$\mu_{12}=$'+str(round(mu, 3))
            
            ax.errorbar(group[key], group.M_I, yerr=std.M_I, fmt='o', capsize=4, label=cap)

        # Plot observational values
        a = 0.25
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax+tol)
        
        #ax.plot(x, -3.96*np.ones(len(x)), linestyle='--', color='k', label=r'$\omega$ Centauri')
        #ax.fill_between(x, -3.96-0.05, -3.96+0.05, color='k', alpha=a)
        #ax.plot(x, -4.027*np.ones(len(x)), linestyle='--', color='orange', label=r'NGC4258')
        #ax.fill_between(x, -4.027-0.055, -4.027+0.055, color='orange', alpha=a)
        #ax.plot(x, -4.047*np.ones(len(x)), linestyle=':', color='royalblue', label=r'LMC (F20)')
        #ax.fill_between(x, -4.047-0.045, -4.047+0.045, color='royalblue', alpha=a)
        ax.plot(x, -3.958*np.ones(len(x)), linestyle=':', color='k', label=r'LMC (Y19)')
        ax.fill_between(x, -3.958-0.046, -3.958+0.046, color='k', alpha=a)
        
        ax.set_xlabel(label)
        ax.set_ylabel('I-Band Magnitude')
        ax.legend(prop={'size': 14}, loc='best')
        if key == 'z':
            ax.set_xscale('log')
        ax.set_xlim(xmin, xmax+tol)
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
