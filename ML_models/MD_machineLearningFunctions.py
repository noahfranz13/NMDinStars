#!/usr/bin/python3.8
"""
Date Created: 4 June 2022
Date Edited: 4 June 2022
Author: Mitchell T Dennis

This file contains the functions needed when the machine learning algorithms in this directory are used.
"""


import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours


def normalise(data, normFile):
    """
    Normalise master data

    INPUTS:
    DATA: The data to be normalised
    
    RETURNS:
    DATA: A true set of data
    ABSMIN: The absolute minimum of the master data set
    MINIMUM: The shifted minimum of the master data set
    MAXIMUM: The shifted maximum of the master data set
    """
    absMin = abs(data.min())
    data += absMin
    maximum = data.max()
    minimum = data.min()
    data = (data - minimum) / (maximum - minimum)
    normFile.write(str(absMin)+'\t'+str(minimum)+'\t'+str(maximum)+'\n')
    return data, absMin, minimum, maximum


def normaliseFromValues(minimum, maximum, absoluteMin, data):
    """
    Take data and normalise it using already known constants.
    
    INPUTS:
    MINIMUM: The shifted minimum of the master data set
    MAXIMUM: The shifted maximum of the master data set
    ABSOLUTEMIN: The absolute minimum of the master data set

    RETURNS:
    DATA: A normalised data point or set of data points (between 0-1)    
    """
    data += absoluteMin
    data = (data - minimum) / (maximum - minimum)
    return data


def deNormalise(minimum, maximum, absoluteMin, data):
    """
    Take normalised (between 0-1) data and transform it using already known constants.
    
    INPUTS:
    MINIMUM: The shifted minimum of the master data set
    MAXIMUM: The shifted maximum of the master data set
    ABSOLUTEMIN: The absolute minimum of the master data set

    RETURNS:
    DATA: A true data point or set of data points
    """
    return (data*(maximum - minimum) + minimum) - absoluteMin


def splitData(data, inputColumns, outputColumns, splitFraction, seed=None, smote=False):
    """
    Split data for machine learning training using SMOTE upsampling.

    INPUTS:
    DATA: A Pandas dataframe containing the data
    INPUTCOLUMNS: The data columns to be used as inputs
    OUTPUTCOLUMNS: The data columns to be used as outputs
    SPLITFRACTION: The ratio of training to testing/validation data
    
    RETURNS:
    XTRAIN: The input training data
    YTRAIN: The output training data
    XTEST: The input testing data
    YTEST: The output testing data
    """
    dataSet = data.sample(frac=1, random_state=seed)
    trainingData = dataSet.head(n=int(np.floor(len(dataSet)*splitFraction)))
    xTrain = trainingData[inputColumns]
    yTrain = trainingData[outputColumns]
    if smote:
        oversample = SMOTE(random_state=seed, k_neighbors=10)
        xTrain, yTrain = oversample.fit_resample(xTrain, yTrain)    
    testingData = dataSet.tail(n=int(np.floor(len(dataSet)*(1-splitFraction))))
    xTest = testingData[inputColumns].head(int(np.floor(len(testingData)/2)))
    yTest = testingData[outputColumns].head(int(np.floor(len(testingData)/2)))
    xVerify = testingData[inputColumns].tail(int(np.ceil(len(testingData)/2)))
    yVerify = testingData[outputColumns].tail(int(np.ceil(len(testingData)/2)))
    return xTrain, yTrain, xTest, yTest, xVerify, yVerify


def createModel(inputDim, outputDim, numLayers, scaling, hiddenInit, dropoutFrac):
    """
    Create a neural network using Keras

    INPUTS:
    INPUTDIM: The number of columns of input data
    OUTPUTDIM: The number of columns of output data
    NUMLAYERS: The number of layers in the neural network
    SCALING: The fraction by which to reduce the number of hidden nodes each layer
    HIDDENINIT: The initial number of hidden nodes
    DROPOUTFRAC: The number of nodes to randomly drop during training

    RETURNS:
    A Keras/TensorFlow neural network
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inputDim, input_dim=inputDim, activation='relu'))
    model.add(tf.keras.layers.Dense(hiddenInit, activation='relu'))
    for i in range(1, numLayers):
        model.add(tf.keras.layers.Dense(int(np.floor(hiddenInit / (scaling * i)) + 1), activation='relu'))
        if i / numLayers >= 0.5:
            model.add(tf.keras.layers.Dropout(dropoutFrac, input_shape=(int(np.floor(hiddenInit / (scaling * i)) + 1),), seed=5478964))
    model.add(tf.keras.layers.Dense(outputDim, activation='sigmoid'))
    return model


def MANA_runtime_loss_function(alpha):
    def custom_loss(y_true, y_pred):
        batch_size = len(y_true)
        errors = y_true - y_pred
        mask = K.less(y_pred, y_true)
        loss = K.switch(mask, ((y_true - y_pred)*(1 - alpha))**2, ((y_true - y_pred)*alpha)**2) 
        return loss
    return custom_loss


# ((y_true - y_pred) + (y_true) + (rerun_scaling*y_true-y_pred)*alpha)**2
