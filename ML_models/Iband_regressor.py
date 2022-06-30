'''
ML regression algorithm to compute the I-Band magnitude of MESA models
given the stellar mass, helium abundance, metallicity, and mu_12
'''
# imports
import os
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError as rms
from tensorflow.keras.metrics import MeanAbsoluteError as mae

from sklearn.metrics import mean_squared_error

from ML_Helpers import *

# read in the data using pandas
datadir = '/home/ubuntu/Documents/NMDinStars/analysis/output_analysis/'
infile = os.path.join(datadir, 'allData.csv')
allData = pd.read_csv(infile) # time is in minutes
# cut out too old, he flash, and didn't converge
allData = allData[allData.flag == 0]
data = allData[['mass', 'y', 'z', 'M_I', 'M_I_err']]

# set hyper parameters
nLayers = 10
epochs = 120
batch = 120
initLR = 1e-4

inNames = ['mass', 'y', 'z']
outNames = ['M_I', 'M_I_err']

# normalize the data
normData, minVal, maxVal = minNormalize(data)

# split up the data
train, val, test = splitData(normData)

# build the model
model = buildModel(nLayers, nOutputs=2,
                   activation='tanh',
                   metrics=[rms(), mae()],
                   optimizer=Adam(learning_rate=initLR))

# train the model
cb = EarlyStopping(monitor='val_loss',
                   min_delta=1e-5,
                   patience=10,
                   restore_best_weights=True)
lrCalib = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.01,
                            patience=5,
                            cooldown=5,
                            verbose=1)

hist = model.fit(train[inNames], train[outNames],
                 validation_data=(val[inNames], val[outNames]),
                 epochs=epochs,
                 batch_size=batch,
                 verbose=0,
                 callbacks=[cb, lrCalib])

# save the model
model.save('IBand.h5')

# plot loss
plotLoss(hist.history, name='loss_regressor.jpeg')

# test the model with predict method
pred = model.predict(test[inNames])

predI, predErr = pred[:,0], pred[:,1]

# denormalize Mitchell's data using his denormalization function
predI_deNorm = inverseMinNormalize(predI, minVal.M_I, maxVal.M_I)
predErr_deNorm = inverseMinNormalize(predErr, minVal.M_I_err, maxVal.M_I_err)
Iband = inverseMinNormalize(test.M_I, minVal.M_I, maxVal.M_I)
Ierr = inverseMinNormalize(test.M_I_err, minVal.M_I_err, maxVal.M_I_err)

plotCompareHist(Iband, predI_deNorm, name='output_2dHist_Iband.jpeg')
plotCompareHist(Ierr, predErr_deNorm, name='output_2dHist_IbandErr.jpeg')


# compute mean squared error
rmse_I = mean_squared_error(Iband, predI_deNorm, squared=False)
rmse_Err = mean_squared_error(Ierr, predErr_deNorm, squared=False)

# print and write metrics to a file

forFile = ''
forFile += f"Loss: {hist.history['val_loss'][-1]}\n"
forFile += f'Root Mean Squared Error on I-Band: {rmse_I}\n'
forFile += f'Root Mean Squared Error on I-Band Error: {rmse_Err}' 

print(forFile)

with open('regressor_metrics.txt', 'w') as f:
    f.write(forFile)
