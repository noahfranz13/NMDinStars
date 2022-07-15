'''
File to run the MCMC for the Neutrino Magnetic Dipole Moment
'''
# imports
import os, sys
import pandas as pd
import numpy as np
import emcee
import corner
from multiprocessing import Pool
from tensorflow.keras.models import load_model

from ML_Helpers import inverseMinNormalize, minNormalize, norm1
from MD_machineLearningFunctions import normaliseFromValues, deNormalise 

import matplotlib.pyplot as plt
import seaborn as sb

sb.set(context='talk', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

# Make sure numpy multiprocessing doesn't happen
os.environ["OMP_NUM_THREADS"] = "1"

# define cwd
cwd = '/home/ubuntu/Documents/NMDinStars/ML_models'

# Declare ML models and any args as global variables
# NOTE: This is a bad practice but necessary to reduce MCMC runtime
# significantly due to the compression practice used in emcee

classifier = None
regressor = None

# define observed values for the I-band
# LMC value from Freedman et al (2020)
obsI = -4.047
obsErr = 0.045

# Get ML errors
IbandErr = None
IerrErr = None

# get normalization constants
const = pd.read_csv(os.path.join(cwd, 'norm_const.txt'), index_col=0)

# constant boolean to know which parts of the script to run
useMu = None

def io():
    '''
    Read in everything we need depending on useMu
    '''
    global classifier
    global regressor
    global IbandErr
    global IerrErr
    
    if useMu:
        classifier = load_model(os.path.join(cwd, 'classifier/classify_mesa.h5'))
        regressor = load_model(os.path.join(cwd, 'regressor/IBand.h5'))
        IbandErr = np.load(os.path.join(cwd,'regressor/Iband_error.npy'))
        IerrErr = np.load(os.path.join(cwd, 'regressor/Ierr_error.npy'))
    else:
        classifier = load_model(os.path.join(cwd, 'SM/classify_mesa_SM.h5'))
        regressor = load_model(os.path.join(cwd, 'SM/IBand_SM.h5'))
        IbandErr = np.load(os.path.join(cwd,'SM/Iband_error.npy'))
        IerrErr = np.load(os.path.join(cwd, 'SM/Ierr_error.npy'))

def norm(theta):
    '''
    Normalizes theta to pass into ML algorithms
    '''
    # normalize the input vector
    if useMu:
        m, y, z, mu = theta
        m = norm1(m, const['min'].loc['mass'], const['max'].loc['mass'])
        y = norm1(y, const['min'].loc['y'], const['max'].loc['y'])
        z = norm1(z, const['min'].loc['z'], const['max'].loc['z'])
        mu = norm1(mu, const['min'].loc['mu'], const['max'].loc['mu'])
        thetaNorm = np.array([m, y, z, mu])[None,:]
    else:
        m, y, z = theta
        m = normaliseFromValues(1.4, 2.9524999999999997, 0.7, m)
        y = normaliseFromValues(0.4, 0.50485, 0.2, y)
        z = normaliseFromValues(2.68878e-05, 0.0406135894, 1.34439e-5, z)
        thetaNorm = np.array([m, y, z])[None,:]        

    return thetaNorm

def ML(theta):
    '''
    Use the ML classiifer and regressor to predict the I-Band
    magnitudes given mass, Y, Z, mu_12
    '''

    # normalize values
    thetaNorm = norm(theta)

    # call the ML models    
    IbandNorm, IerrNorm = regressor(thetaNorm).numpy()[0]

    if useMu:
        # denormalize Iband and Ierr
        Iband = inverseMinNormalize(IbandNorm,
                                    const['min'].loc['M_I'],
                                    const['max'].loc['M_I'])
        Ierr = inverseMinNormalize(IerrNorm,
                                   const['min'].loc['M_I_err'],
                                   const['max'].loc['M_I_err'])
    else:
        # denormalize Iband and Ierr
        Iband = deNormalise(0.0, 5.6129999999999995, 5.611, IbandNorm)
        Ierr = deNormalise(0.106, 0.8350000000000001, 0.053, IerrNorm)
        
    return Iband, Ierr

def logLikelihood(theta):
    '''
    likelihood function of a given point in the parameter space

    theta [vector] : list of ML model inputs
    obsI [float] : Observed I-band value to compare to
    obsErr [float] : error on the observed I-band value
    '''
    Iband, Ierr = ML(theta)

    # propagate ML uncertainties
    Iband = Iband + np.random.choice(IbandErr)
    Ierr = Ierr + np.random.choice(IerrErr)
    
    err2 = Ierr**2 + obsErr**2 # add err in quadrature
    # return the max likelihood function
    return -0.5 * ((obsI-Iband)**2 / err2 + np.log(2*np.pi*err2))
    
def logPrior(theta):
    '''
    Prior to pass to log_prob to be passed into the MCMC

    theta [list] : list of the ML model inputs (m, y, z, mu)
    '''
    if useMu:
        m, y, z, mu = theta
    else:
        m, y, z = theta
        mu = 1
        
    # call the classification algorithm and check flag
    thetaNorm = norm(theta) # normalize to pass into ML
    
    flag = classifier(thetaNorm).numpy()
    flag = np.argmax(flag)
    if flag != 0:
        return -np.inf

    # other priors
    if 0.7 <= m <= 2.25 and 0.2 <= y <= 0.3 and 1e-5 <= z <= 0.04 and 0 <= mu <= 4:
        mPrior = 0 #-2.35*np.log(m) # Use Salpeter IMF
        yPrior = 0 # constant
        zPrior = 0 # constant
        if len(theta) > 3:
            muPrior = 0 # constant
            return mPrior + yPrior + zPrior + muPrior
        else:
            return mPrior + yPrior + zPrior
    else:
        return -np.inf

def logProb(theta):
    '''
    PDF to be passed into emcee EnsembleSampler Class

    theta [vector] : list of ML model inputs
    obsI [float] : Observed I-band value to compare to
    obsErr [float] : error on the observed I-band value

    '''
    prior = logPrior(theta)
    likelihood = logLikelihood(theta)
    
    if not np.isfinite(prior):
        return -np.inf
    return prior + likelihood

def main():
    '''
    Run the MCMC using the helper functions defined in this script
    '''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-mu', dest='useMu', action='store_false')
    parser.set_defaults(useMu=True)
    args = parser.parse_args()

    global useMu
    useMu = args.useMu
    io() # read in stuff we need

    nsteps = 5000 #500000
    nwalkers = 32
    
    if useMu:
        # run the MCMC
        ndim = 4
        initPos = [1.5, 0.25, 0.01, 1] + 1e-4 * np.random.randn(nwalkers, ndim)
        labels = ['Mass', 'Y', 'Z', r'$\mu_{12}$']
        
    else:
        ndim = 3
        initPos = [1.5, 0.25, 0.01] + 1e-4 * np.random.randn(nwalkers, ndim)
        labels = ['Mass', 'Y', 'Z']

    print(f'Using {ndim} dimensions')
    # output hdf5 file to save progress
    back = emcee.backends.HDFBackend('nmdm.h5')
    back.reset(nwalkers, ndim)

    autocorr = []
    indexes = []
    idx = 0
    isConverged = False
    oldTau = np.inf
    mod = nsteps/100
    
    # Run the MCMC
    with Pool() as p:
        es = emcee.EnsembleSampler(nwalkers,
                                   ndim,
                                   logProb,
                                   pool=p,
                                   backend=back)

        # sampler.run_mcmc(initPos, nsteps, progress=True)
        for _ in es.sample(initPos, iterations=nsteps, progress=True):

            if not es.iteration % mod:
                # record autocorrelation time
                tau = es.get_autocorr_time(tol=0, discard=es.iteration//2)
                autocorr.append(tau)
                indexes.append(idx)
                
                # check convergence if model hasn't converged
                isConverged = np.all(tau*100 < es.iteration)
                isConverged &= np.all(np.abs(oldTau-tau)/tau < 0.01)
                if isConverged:
                    print(f"Model converged at step: {es.iteration}")

                oldTau = tau

            if not es.iteration % (mod*10):
                # make a corner plot
                flatSamples = es.get_chain(discard=es.iteration//2, flat=True)
                fig = corner.corner(flatSamples,
                                    labels=labels)
                fig.savefig(f"corner_i{es.iteration}.jpeg",
                            bbox_inches='tight',
                            transparent=False)   
            idx += 1

    autocorr = np.array(autocorr)
    indexes = np.array(indexes)
    
    # save outputs
    flatSamples = es.get_chain(discard=es.iteration//2, flat=True)
    np.save('chain', flatSamples)
    np.save('autocorr', autocorr)
    np.save('indexes', indexes)
    
    # Make corner plot of the outputs
    fig = corner.corner(flatSamples, labels=labels)
    fig.savefig("corner_final.jpeg", bbox_inches='tight', transparent=False)

    # Plot auto correlation time vs. iteration
    fig, ax = plt.subplots(1, figsize=(8,6))

    for label, a in zip(labels, autocorr.T):
        ax.plot(indexes, a, label=label)

    ax.set_xlabel('MCMC Iterations')
    ax.set_ylabel('Auto Correlation Time')
    ax.legend()
        
    fig.savefig("auto_correlation.jpeg", bbox_inches='tight', transparent=False)            
    
if __name__ == '__main__':
    sys.exit(main())
