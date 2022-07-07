# Write some useful functions for all ML algorithms
# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
    
import seaborn as sb
sb.set(context='talk', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

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

def norm1(d, minVal, maxVal):
    '''
    Normalizes one data vector use min normalize method

    d [float/int] : data point to normalize 
    MINVAL [FLOAT] : MINIMUM VALUE USED IN TRAINING
    MAXVAL [FLOAT] : MAXIMUM VALUE USED IN TRAINING
    '''
    return (d - minVal) / (maxVal - minVal), minVal, maxVal

def inverseMinNormalize(dNorm, minVal, maxVal):
    '''
    Denormalized ML outputs
    
    dNorm [np.array] : dataset to inverse normalize
    return : inversed normalize the data
    '''
    return dNorm * (maxVal - minVal) + minVal

# define a useful splitting function
def splitData(df, trainFrac=0.8, valFrac=0.10, testFrac=0.10, seed=None):
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
def buildModel(nLayers, activation='relu', loss='mse', optimizer='adam', metrics=[], nOutputs=2, nInputs=4, actLast='sigmoid'):
    '''
    Builds the Sequential model for ML
    Note: make this more complex with more options as needed
    
    nLayers [int] : number of dense layers
    activation [String] : activation function to use in each Dense layer
    loss [String] : loss function to use during compiling
    optimizer [String] : optimizer to use for training, default is adam
    metrics [List] : metrics to return from the model training
    nOutputs [int] : number of output values, used to set # of nodes in last layer
    actLast [String] : activation function to use in the last layer
    
    returns : a Keras Sequential Model object
    '''
    
    # initialize the model
    model = Sequential()

    nodes = 2**nLayers
    # start large than get smaller
    while nLayers > 0:
        if nLayers == 1:
            model.add(Dense(nOutputs, activation='sigmoid'))
        else:
            model.add(Dense(nodes, activation=activation))
        
        nodes = nodes/2
        nLayers -= 1
        
        
    # compile the model
    model.compile(loss=loss,
                 optimizer=optimizer,
                 metrics=metrics)
        
    return model

def plotLoss(h, key='loss', name='loss.jpeg'):
    '''
    Plots the loss of the Keras history dict
    '''
    fig, ax = plt.subplots()

    ax.plot(h[key], label='Training')
    ax.plot(h['val_' + key], label='Validation')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(name, bbox_inches='tight', transparent=False)

def plotAccuracy(h, key='accuracy', name='accuracy.jpeg'):
    '''
    Plots the loss of the Keras history dict
    '''
    fig, ax = plt.subplots()

    ax.plot(h[key], label='Training')
    ax.plot(h['val_' + key], label='Validation')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig(name, bbox_inches='tight', transparent=False)

def plotFalseNegPos(h):
    '''
    Plots the loss of the Keras history dict
    '''
    # plot false negatives
    fig, ax = plt.subplots()

    ax.plot(h['false_negatives'], label='Training')
    ax.plot(h['val_false_negatives'], label='Validation')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('False Negative')
    ax.legend()
    fig.savefig('FalseNegative.jpeg', bbox_inches='tight', transparent=False)

    # plot false positives
    fig, ax = plt.subplots()

    ax.plot(h['false_positives'], label='Training')
    ax.plot(h['val_false_positives'], label='Validation')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('False Positive')
    ax.legend()
    fig.savefig('FalsePositive.jpeg', bbox_inches='tight', transparent=False)


def plotCompareHist(actual, prediction, name='output_2Dhist.jpeg'):
    '''
    Plots a 2D histogram of the predicted ouput vs. the expected
    output

    actual [array]     : expected array of outputs
    prediction [array] : models predicted outputs

    Returns : Nothing, saves a figure
    '''
    import cmasher as cm
    
    fig, ax = plt.subplots()

    count, xedge, yedge, im = ax.hist2d(actual, prediction,
                                        bins=20,
                                        cmap=cm.dusk)
    ax.set_xlabel('Expected Results')
    ax.set_ylabel('Model Prediction')
    ax.set_aspect('equal')
    ax.tick_params(axis='both',
                   which='major',
                   direction='in',
                   length=5,
                   color='white')
    fig.colorbar(im, label='Number of Models')
    fig.savefig(name,
                bbox_inches='tight',
                transparent=False)
    
