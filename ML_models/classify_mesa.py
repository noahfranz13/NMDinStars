'''
ML classifier algorithm to differentiate between good mesa models
and models that are too old or HE shell flash
'''
# imports
import os
import pandas as pd

from ML_Helpers import *

# read in the data using pandas
datadir = '/home/ubuntu/Documents/NMDinStars/mesa_mod/mesa-r12778/star/test_suite/NeutrinoMagneticDipoleMesa/WorthyLeeBC'
infile = os.path.join(datadir, 'postProcess_output_mesa-1296.txt')
data = pd.read_csv(infile, index_col=0) # time is in minutes
data = data[data['flag'] != 1] # cut out flag = 1

# set hyper parameters
nLayers = 6
epochs = 10
batch = 10

# normalize the data
normData, minVal, maxVal = minNormalize(data[['flag', 'mass', 'y', 'z', 'mu']])

# randomly split the data
train, val, test = splitData(normData)

# generate the ML model
model = buildModel(nLayers, metrics=['accuracy'])

# train the model
print('Training the model...')
hist = model.fit(train[['mass', 'y', 'z', 'mu']], train.flag,
                 validation_data=(val[['mass', 'y', 'z', 'mu']], val.flag),
                 epochs=epochs,
                 batch_size=batch,
                 verbose=0)

plot(hist.history)
plot(hist.history, key='accuracy')

# test the model
loss, acc = model.evaluate(test[['mass', 'y', 'z', 'mu']], test.flag)
print('Testing Accuracy: ', acc)
