
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
    
    fig = corner.corner(chain,
                        labels=['Mass', 'Y', r'log$_{10}$(Z)', r'$\mu_{12}$'],
                        show_titles=True,
                        color='dodgerblue',
                        smooth=True,
                        plot_datapoints=False,
                        fill_contours=True)
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

def CI(chain, alpha):
    '''
    Calculate the confidence intervals around the median
    of each input parameter in the chain
    
    chain [Array] : np array of output samples from the MCMC
    alpha [float] : confidence level to compute
    center [string] : techniique to get central value of distribution
                      that the CI is based around. Supported options are
                      'median', 'mode', 'mean'
    '''
    
    forFile = 'Confidence Intervals: \n\n'
    labels = ['M', 'Y', 'Z', 'mu']
    for col, label in zip(chain.T, labels):

        middle = 0.5
        q1 = np.quantile(col, middle-(alpha/2))
        q2 = np.quantile(col, middle)
        q3 = np.quantile(col, middle+(alpha/2))
        
        forFile += f'{label} {alpha*100}% CI:\n'
        forFile += f'         Q1: {q1}\n'
        forFile += f'Q2 (median): {q2}\n'
        forFile += f'         Q3: {q3}\n'
        forFile += f'{label} = {q2} (+{q3-q2}, -{q2-q1})'
        forFile += '\n'
        
    print(forFile)
    with open(f'CI_{alpha}.txt', 'w') as f:
        f.write(forFile)
    
def plot2Dhists(chain):
    '''
    Plot 2D histograms for mu vs. other parameters
    '''
    labels = ['M', 'Y', 'Z']
    mu = chain.T[-1]
    for col, label in zip(chain.T[:-1], labels):
        print(len(mu), len(col))
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist2d(col, mu, bins=20)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\mu_{12}$')
        fig.savefig(f"mu_vs_{label}_2dhist.jpeg", bbox_inches='tight', transparent=False)

def plotHists(chain):
    '''
    Plot histograms for input parameters
    '''
    labels = ['M', 'Y', 'Z', 'mu12']
    for col, label in zip(chain.T, labels):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(col, bins=20)
        ax.set_xlabel(label)
        ax.set_ylabel('N')
        fig.savefig(f"{label}_hist.jpeg", bbox_inches='tight', transparent=False)
        
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='plotLogged', action='store_true')
    parser.set_defaults(plotLogged=False)
    args = parser.parse_args()
    
    chain = np.load('chain.npy')
    
    # log Z
    if args.plotLogged:
        chain[:,2] = np.log10(chain[:, 2])
    
    plotCorner(chain)
    testGaussian(chain)
    CI(chain, 0.68)
    CI(chain, 0.995)
    plot2Dhists(chain)
    plotHists(chain)
        
if __name__ == '__main__':
    sys.exit(main())
