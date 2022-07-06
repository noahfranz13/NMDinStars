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

def readML():
    '''
    Read in ML models
    '''
    classifier = load_model('/home/ubuntu/Documents/NMDinStars/ML_models/classify_mesa.h5')
    regressor = load_model('/home/ubuntu/Documents/NMDinStars/ML_models/IBand.h5')

    return classifier, regressor
  

def runML(theta, classify, regressor):
    '''
    Use the ML classiifer and regressor to predict the I-Band
    magnitudes given mass, Y, Z, mu_12
    '''

    flag = classify(theta).numpy()
    Iband = regressor(theta).numpy()
    return Iband, flag

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

    # read in the ML algorithms just once
    classify, reg = readML()
    
    print(classify, Iband)

    

if __name__ == '__main__':
    sys.exit(main())
