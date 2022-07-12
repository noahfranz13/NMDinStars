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

# Declare ML models and any args as global variables
# NOTE: This is a bad practice but necessary to reduce MCMC runtime
# significantly due to the compression practice used in emcee

classifier = load_model('/home/nfranz/NMDinStars/ML_models/SM/classify_mesa_SM.h5')
regressor = load_model('/home/nfranz/NMDinStars/ML_models/SM/IBand_SM.h5')

# define observed values for the I-band
# LMC value from Freedman et al (2020)
obsI = -4.047
obsErr = 0.045

# Get ML errors
IbandErr = np.load('/home/nfranz/NMDinStars/ML_models/SM/Iband_error.npy')
IerrErr = np.load('/home/nfranz/NMDinStars/ML_models/SM/Ierr_error.npy')

def ML(theta):
    '''
    Use the ML classiifer and regressor to predict the I-Band
    magnitudes given mass, Y, Z, mu_12
    '''
    # normalize the input vector
    m, y, z = theta
    
    m = normFromValues(1.4, 2.9524999999999997, 0.7, m)
    y = normFromValues(0.4, 0.50485, 0.2, y)
    z = normFromValues(2.68878e-05, 0.0406135894, 1.34439e-5, z)
    thetaNorm = np.array([m, y, z])[None,:]

    # call the ML models
    flag = classifier(thetaNorm).numpy()
    
    flag = np.argmax(flag)
    
    if flag != 0:
        return -np.inf, -np.inf, flag
    
    IbandNorm, IerrNorm = regressor(thetaNorm).numpy()[0]
    
    # denormalize Iband and Ierr
    Iband = deNormalise(0.0, 5.6129999999999995, 5.611, IbandNorm)
    Ierr = deNormalise(0.106, 0.8350000000000001, 0.053, IerrNorm)
    
    return Iband, Ierr, flag

def logLikelihood(theta):
    '''
    likelihood function of a given point in the parameter space

    theta [vector] : list of ML model inputs
    obsI [float] : Observed I-band value to compare to
    obsErr [float] : error on the observed I-band value
    '''
    Iband, Ierr, flag = ML(theta)

    # propagate ML uncertainties
    Iband = Iband + np.random.choice(IbandErr)
    Ierr = Ierr + np.random.choice(IerrErr)
    
    if np.isfinite(Iband):    
        err2 = Ierr**2 + obsErr**2 # add err in quadrature
        # return the max likelihood function
        return -0.5 * ((obsI-Iband)**2 / err2 + np.log(2*np.pi*err2))
    else:
        return -np.inf

def logPrior(theta):
    '''
    Prior to pass to log_prob to be passed into the MCMC

    theta [list] : list of the ML model inputs (m, y, z)
    '''
    m, y, z = theta

    if 0.7 <= m <= 2.25 and 0.2 <= y <= 0.3 and 1e-5 <= z <= 0.04:
        mPrior = -2.35*np.log(m) # Use Salpeter IMF
        yPrior = 0 # constant
        zPrior = 0 # constant
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
    
    if not np.isfinite(prior) or not np.isfinite(likelihood):
        return -np.inf
    return prior + likelihood

def main():
    '''
    Run the MCMC using the helper functions defined in this script
    '''

    # run the MCMC
    nwalkers = 32
    ndim = 4
    nsteps = 500000
    initPos = [1.5, 0.25, 0.01, 1] + 1e-4 * np.random.randn(nwalkers, ndim)

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
                tau = es.get_autocorr_time(tol=0)
                autocorr.append(tau)
                indexes.append(idx)
                
                # check convergence if model hasn't converged
                if not isConverged:
                    isConverged = np.all(tau*100 < es.iteration)
                    isConverged &= np.all(np.abs(oldTau-tau)/tau < 0.01)
                    if isConverged:
                        print(f"Model converged at step: {es.iteration}")

                oldTau = tau

            if not es.iteration % (mod*10):
                # make a corner plot
                flatSamples = es.get_chain(discard=100, flat=True)
                fig = corner.corner(flatSamples,
                                    labels=['Mass', 'Y', 'Z'])
                fig.savefig(f"corner_i{es.iteration}.jpeg",
                            bbox_inches='tight',
                            transparent=False)   
            idx += 1

    autocorr = np.array(autocorr)
    indexes = np.array(indexes)
    
    # save outputs
    flatSamples = es.get_chain(discard=100, flat=True)
    np.save('chain', flatSamples)
    np.save('autocorr', autocorr)
    np.save('indexes', indexes)
    
    # Make corner plot of the outputs
    fig = corner.corner(flatSamples, labels=['Mass', 'Y', 'Z'])
    fig.savefig("corner_final.jpeg", bbox_inches='tight', transparent=False)

    # Plot auto correlation time vs. iteration
    fig, ax = plt.subplots(1, figsize=(8,6))
    labels = ['m', 'y', 'z']
    
    for label, a in zip(labels, autocorr.T):
        ax.plot(indexes, a, label=label)

    ax.set_xlabel('MCMC Iterations')
    ax.set_ylabel('Auto Correlation Time')
    ax.legend()
        
    fig.savefig("auto_correlation.jpeg", bbox_inches='tight', transparent=False)            
    
if __name__ == '__main__':
    sys.exit(main())
