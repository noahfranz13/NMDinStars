'''
File to run the MCMC for the Neutrino Magnetic Dipole Moment
'''
# imports
import os, sys
import pandas as pd
import numpy as np
import emcee
import corner
import multiprocessing as mp
from scipy.stats import chisquare
from tensorflow.keras.models import load_model

#sys.path.append('/home/ubuntu/Documents/NDMinStars/ML_models')
from ML_Helpers import inverseMinNormalize, minNormalize, norm1

# Declare ML models as global variables
# NOTE: This is a bad practice but necessary to reduce MCMC runtime
# significantly due to the model compression practice used in emcee

classifier = load_model('/home/ubuntu/Documents/NMDinStars/ML_models/classify_mesa.h5')
regressor = load_model('/home/ubuntu/Documents/NMDinStars/ML_models/IBand.h5')  

def ML(theta):
    '''
    Use the ML classiifer and regressor to predict the I-Band
    magnitudes given mass, Y, Z, mu_12
    '''
    # read in normalization constants
    constReg = pd.read_csv('/home/ubuntu/Documents/NMDinStars/ML_models/regression_norm_const.txt', index_col=0)
    constClass = pd.read_csv('/home/ubuntu/Documents/NMDinStars/ML_models/classify_norm_const.txt', index_col=0)

    # normalize the input vector
    m, y, z, mu = theta
    
    mClass = norm1(m, constClass['min'].loc['mass'], constClass['max'].loc['mass'])
    yClass = norm1(m, constClass['min'].loc['y'], constClass['max'].loc['y'])
    zClass = norm1(m, constClass['min'].loc['z'], constClass['max'].loc['z'])
    muClass = norm1(m, constClass['min'].loc['mu'], constClass['max'].loc['mu'])
    
    mReg = norm1(m, constReg['min'].loc['mass'], constReg['max'].loc['mass'])    
    yReg = norm1(m, constReg['min'].loc['y'], constReg['max'].loc['y'])    
    zReg = norm1(m, constReg['min'].loc['z'], constReg['max'].loc['z'])    
    muReg = norm1(m, constReg['min'].loc['mu'], constReg['max'].loc['mu'])    

    # call the ML models 
    flag = classifier(np.array([mClass, yClass, zClass, muClass])[None,:]).numpy()

    if flag != 0:
        return -np.inf, -np.inf, flag
    
    IbandNorm, IerrNorm = regressor(np.array([mReg, yReg, zReg, muReg])[None,:]).numpy()
    
    # denormalize Iband and Ierr
    Iband = inverseMinNormalize(Iband,
                                normConst['min'].loc['M_I'],
                                normConst['max'].loc['M_I'])
    Ierr = inverseMinNormalize(Ierr,
                               normConst['min'].loc['M_I_err'],
                               normConst['max'].loc['M_I_err'])
    
    return Iband, Ierr, flag

def logLikelihood(theta, obsI, obsErr):
    '''
    likelihood function of a given point in the parameter space

    theta [vector] : list of ML model inputs
    obsI [float] : Observed I-band value to compare to
    obsErr [float] : error on the observed I-band value
    '''
    Iband, Ierr, flag = ML(theta)

    if np.isfinite(Iband):
    
        err2 = Ierr**2 + obsErr**2 # add err in quadrature
        # return the max likelihood function
        return -0.5 * (obsI-Iband)**2 / err2 + np.log(2*np.pi*err2)
    else:
        return -np.inf

def logPrior(theta):
    '''
    Prior to pass to log_prob to be passed into the MCMC

    theta [list] : list of the ML model inputs (m, y, z, mu)
    '''
    m, y, z, mu = theta

    if 0.7 < m < 2.25 and 0.2 < y < 0.3 and 1e-5 < z < 0.04 and 1e-2 < mu < 4:
        prior = 0 # for now just use constant prior for everything
        return prior
    else:
        return -np.inf

def logProb(theta, obsI, obsErr):
    '''
    PDF to be passed into emcee EnsembleSampler Class

    theta [vector] : list of ML model inputs
    obsI [float] : Observed I-band value to compare to
    obsErr [float] : error on the observed I-band value

    '''
    prior = logPrior(theta)
    likelihood = logLikelihood(theta, obsI, obsErr)
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
    nsteps = 5000
    initPos = [1.5, 0.25, 0.01, 1] + 1e-4 * np.random.randn(nwalkers, ndim)

    # define observed values for the I-band
    # LMC value from Freedman et al (2020)
    obsI = -4.047
    obsErr = 0.045

    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    logProb, args=(obsI, obsErr))
    sampler.run_mcmc(initPos, nsteps, progress=True)

    # Make some plots of the outputs
    print(sampler.get_autocorr_time())

    flatSamples = sampler.get_chain(discard=100, thin=15, flat=True)
    corner.corner(flatSamples, labels=['Mass', 'Y', 'Z', r'$\mu_{12}$'])

if __name__ == '__main__':
    sys.exit(main())
