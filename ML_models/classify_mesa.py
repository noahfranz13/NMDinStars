'''
ML classifier algorithm to differentiate between good mesa models
and models that are too old or HE shell flash
'''
# imports
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from ML_Helpers import *

sb.set(context='talk', palette='Set1', style='whitegrid')

# read in the data using pandas
datadir = '/home/ubuntu/Documents/learnML'
infile = os.path.join(datadir, 'runtimesHost.txt')
times = pd.read_csv(infile, index_col=False, names=['flag', 'm', 'y', 'z', 'mu', 't'], sep='\t') # time is in minutes
times = times[times['flag'] != 1] # cut out flag = 1
print(times[times.flag == 2])

# set hyper parameters
nLayers = 5
epochs = 10
batch = 10

# normalize the data
normData, minVal, maxVal = minNormalize(times)
