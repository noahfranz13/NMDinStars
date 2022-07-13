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

sb.set(context='talk', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

def plotCorner():
    '''
    Create a prettier corner plot
    '''

    flatSamples = np.load('chain.npy')

    fig = corner.corner(flatSamples,
                        labels=['Mass', 'Y', 'Z', r'$\mu_{12}$'],
                        show_titles=True)
    fig.savefig("corner_pretty.jpeg", bbox_inches='tight', transparent=False)

    
def main():

    plotCorner()
        
if __name__ == '__main__':
    sys.exit(main())
