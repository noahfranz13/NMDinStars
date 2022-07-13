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

def plotCorner(chain):
    '''
    Create a prettier corner plot
    '''
    # log Z
    chain[:, 2] = np.log10(chain[:, 2])
    
    fig = corner.corner(chain,
                        labels=['Mass', 'Y', r'log$_{10}$(Z)', r'$\mu_{12}$'],
                        show_titles=True,
                        color='dodgerblue',
                        smooth=True,
                        plot_datapoints=False,
                        fill_contours=True,
                        title_quantiles=[0.68])
    fig.savefig("corner_pretty.jpeg", bbox_inches='tight', transparent=False)

def testGaussian(chain):
    '''
    Test if the output distributions are gaussian
    '''
    from scipy.stats import kstest, norm

    # implement the Kolmogorov-Smirnov p-test
    labels = ['M', 'Y', 'Z', 'mu']
    forFile = 'Kolmogorov-Smirnov Normality Test\n'
    forFile += 'p > 0.05 : Gaussian\n\n'
    for col, label in zip(chain.T, labels):
        stat, p = kstest(col, 'norm')
        forFile += f'{label} p-value: {p}\n'
        if p > 0.05:
            forFile += 'Normal Distribution!\n'
        else:
            forFile += 'Non-Normal Distribution\n'
        forFile += '\n'
        
    print(forFile)
    with open('normality_test.txt', 'w') as f:
        f.write(forFile)
    
def main():

    chain = np.load('chain.npy')

    plotCorner(chain)
    testGaussian(chain)
        
if __name__ == '__main__':
    sys.exit(main())
