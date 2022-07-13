'''
Analysis script of the mcmc outputs to create nicer plots and get
constraints on input parameters
'''
# imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import corner

sb.set(context='paper', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

def plotCorner():
    '''
    Create a prettier corner plot
    '''

    chain = np.load('chain.npy')

    # log Z
    chain[:, 2] = np.log10(chain[:, 2])
    
    fig = corner.corner(chain,
                        labels=['Mass', 'Y', 'Z', r'$\mu_{12}$'],
                        show_titles=True,
                        color='dodgerblue',
                        smooth=True,
                        plot_datapoints=False,
                        fill_contours=True)
    fig.savefig("corner_pretty.jpeg", bbox_inches='tight', transparent=False)

    
def main():

    plotCorner()
        
if __name__ == '__main__':
    sys.exit(main())
