# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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

# read in the two error files
Iband = np.load('Iband_error.npy')
Ierr = np.load('Ierr_error.npy')

# plot the histograms

def pltHist(err, xlabel):
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.hist(err, bins=25, color='cornflowerblue')
    ax.set_ylabel('N')
    ax.set_xlabel(xlabel)
    fig.savefig(f'{xlabel}_hist.jpeg', transparent=False,
                bbox_inches='tight')

pltHist(Iband, 'Error on I-Band Regression')
pltHist(Ierr, 'Error on I-Band Error Regression')

# compute mean and sd of each distribution
Iband_mean = np.mean(Iband)
Iband_sd = np.std(Iband)
Ierr_mean = np.mean(Ierr)
Ierr_sd = np.std(Ierr)

with open('stats.txt', 'w') as f:
    f.write(f'Iband error Mean: {Iband_mean}\n')
    f.write(f'Iband error std: {Iband_sd}\n')
    f.write(f'Ierr error mean: {Ierr_mean}\n')
    f.write(f'Ierr error std: {Ierr_sd}\n')
