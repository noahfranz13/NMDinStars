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

from MD_machineLearningFunctions import deNormalise
from ML_Helpers import *

# read in the data using pandas
datadir = '/home/ubuntu/Documents/NMDinStars/ML_models/'
infile = os.path.join(datadir, 'fulldata.txt')
allData = pd.read_csv(infile) # time is in minutes
# cut out too old, he flash, and didn't converge
allData = allData[allData.flag == 0]
data = allData[['mass', 'Y', 'Z', 'IBand', 'Ierr']]

# set hyper parameters
nLayers = 10
epochs = 120
batch = 120
initLR = 1e-4

inNames = ['mass', 'Y', 'Z']
outNames = ['IBand', 'Ierr']

# normalize the data
#normData, minVal, maxVal = minNormalize(data)
normData = data # just because the input is already normalized

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
                            patience=3,
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
predI_deNorm = deNormalise(0.0, 5.6129999999999995, 5.611, predI)
predErr_deNorm = deNormalise(0.106, 0.8350000000000001, 0.053, predErr)
Iband = deNormalise(0.0, 5.6129999999999995, 5.611, test.IBand)
Ierr = deNormalise(0.106, 0.8350000000000001, 0.053, test.Ierr)

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
