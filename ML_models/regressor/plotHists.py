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
plt.rcParams['ytick.right'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'


# read in the two error files
Iband = np.load('Iband_error.npy')
Ierr = np.load('Ierr_error.npy')
VI = np.load('VI_error.npy')
VIerr = np.load('VIerr_error.npy')

# plot the histograms

def pltHist(err, xlabel):
    
    fig = sb.displot(kind='kde', data=err, color='cornflowerblue')
    
    
    #fig, ax = plt.subplots(1, figsize=(8,6))
    #ax.hist(err, bins=25, color='cornflowerblue')
    fig.set_ylabels('N')
    fig.set_xlabels(xlabel)
    
    fig.savefig(f'{xlabel}_hist.jpeg', transparent=False,
                bbox_inches='tight')
    
pltHist(Iband, 'Error on I-Band Regression Prediction')
pltHist(Ierr, 'Error on I-Band Error Regression Prediction')
pltHist(VI, 'Error on V-I Regression Prediction')
pltHist(VIerr, 'Error on V-I Error Regression Prediction')

# compute mean and sd of each distribution
Iband_mean = np.mean(Iband)
Iband_sd = np.std(Iband)
Ierr_mean = np.mean(Ierr)
Ierr_sd = np.std(Ierr)
VI_mean = np.mean(VI)
VI_sd = np.std(VI)
VIerr_mean = np.mean(VIerr)
VIerr_sd = np.std(VIerr)

with open('stats.txt', 'w') as f:
    f.write(f'Iband error Mean: {Iband_mean}\n')
    f.write(f'Iband error std: {Iband_sd}\n')
    f.write(f'Ierr error mean: {Ierr_mean}\n')
    f.write(f'Ierr error std: {Ierr_sd}\n')
    f.write(f'V-I error Mean: {VI_mean}\n')
    f.write(f'V-I error std: {VI_sd}\n')
    f.write(f'V-I error error mean: {VIerr_mean}\n')
    f.write(f'V-I error error std: {VIerr_sd}\n')
