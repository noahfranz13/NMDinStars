#!/usr/bin/python3.8
"""
Date Created: 4 June 2022
Date Edited: 4 June 2022
Author: Mitchell T Dennis

This file contains the functions needed when the machine learning algorithms in this directory are used.
"""


import numpy as np

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
