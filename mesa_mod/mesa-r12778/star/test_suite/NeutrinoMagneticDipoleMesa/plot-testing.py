# plot up the testing data

from MesaOutput import MesaOutput

import matplotlib.pyplot as plt
import os

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

    dirpath = '/home/nfranz/research/data/mesa-testing'

    mesa = MesaOutput(dirpath)

    mesa.data = [mesa.data[1], mesa.data[2], mesa.data[3], mesa.data[0]]
    
    fig, ax = plt.subplots(figsize=(16,12))
    for m in mesa:

        Ls = os.path.split(m.file_name)[-1].split('_')
        L = 'Z=' + Ls[3][1:] + r', $\mu_{12}$=' + Ls[4][1:].split('.')[0]
        
        plot(m, ax, idx=90, label=L, linewidth=7, alpha=0.8)

    ax.legend()
    fig.savefig('testing-plot.png', transparent=False, bbox_inches='tight')

if __name__ == '__main__':
    main()
