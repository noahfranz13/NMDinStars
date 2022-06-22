# Write some useful functions for all ML algorithms
# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout # import dropout for further extension of the function
    


# define a normalization function
def minNormalize(d):
    '''
    Normalizes data between 0 and 1 for ML algorithms
    
    d [np.array] : dataset to normalize
    return : normalized dataset, minimum of dataset, maximum of dataset
    '''
    minVal = np.min(d)
    maxVal = np.max(d)
    return (d - minVal) / (maxVal - minVal), minVal, maxVal

def inverseMinNormalize(dNorm, minVal, maxVal):
    '''
    Denormalized ML outputs
    
    dNorm [np.array] : dataset to inverse normalize
    return : inversed normalize the data
    '''
    return dNorm * (maxVal - minVal) + minVal

# define a useful splitting function
def splitData(df, trainFrac=0.7, valFrac=0.15, testFrac=0.15, seed=None):
    '''
    Randomly splits a dataset into training, validation, and testing
    
    df [DataFrame]    : Pandas DataFrame of value for the ML model
    trainFrac [float] : fraction of data (between 0 and 1) to train model with
    valFrac [float]   : fraction of data (between 0 and 1) to validate model training with
    testFrac [float]  : fraction of data (between 0 and 1) to test the model with
    seed [int]        : random seed for reproducability, default is None
    
    returns           : Randomly split dataframe in order of training, validation, testing
    '''
    
    if trainFrac + valFrac + testFrac != 1:
        raise Exception('Your input fractions do not add up to 1!! Exiting...')
        return
    
    # randomly shuffle the dataframe
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # split into three sets
    
    idx1 = int(trainFrac*len(df))
    idx2 = idx1 + int(valFrac*len(df))
    
    train = df.iloc[0:idx1]
    val = df.iloc[idx1:idx2]
    test = df.iloc[idx2:]
    
    return train, val, test

# define a function to build a model given some inputs
def buildModel(nLayers, activation='relu', loss='mse', optimizer='adam', metrics=[]):
    '''
    Builds the Sequential model for ML
    Note: make this more complex with more options as needed
    
    nLayers [int] : number of dense layers
    activation [String] : activation function to use in each Dense layer
    loss [String] : loss function to use during compiling
    optimizer [String] : optimizer to use for training, default is adam
    metrics [List] : metrics to return from the model training
    
    returns : a Keras Sequential Model object
    '''
    
    # initialize the model
    model = Sequential()
    
    # start large than get smaller
    nodes = 2**nLayers
    while nLayers > 0:
        model.add(Dense(nodes, activation=activation))
        
        nodes = nodes/2
        nLayers -= 1
        
    # compile the model
    model.compile(loss=loss,
                 optimizer=optimizer,
                 metrics=metrics)
        
    return model

def plotLoss(h, key='loss'):
    '''
    Plots the loss of the Keras history dict
    '''
    fig, ax = plt.subplots()

    ax.plot(h[key], label='Training')
    ax.plot(h['val_' + key], label='Validation')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(key)
    ax.legend()
        
    
