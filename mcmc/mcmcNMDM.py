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

def readML():
    '''
    Read in ML models
    '''
    pass

def runML(theta):
    '''
    Use the ML classiifer and regressor to predict the I-Band
    magnitudes given mass, Y, Z, mu_12
    '''
    pass

def log_likelihood(theta):
    '''
    likelihood function of a given point in the parameter space
    '''
    pass

def log_prior(theta):
    '''
    Prior to pass to log_prob to be passed into the MCMC
    '''
    pass

def log_prob(theta):
    '''
    PDF to be passed into emcee EnsembleSampler Class
    '''
    pass


def main():
    '''
    Run the MCMC using the helper functions defined in this script
    '''
    pass
    

if __name__ == '__main__':
    sys.exit(main())
