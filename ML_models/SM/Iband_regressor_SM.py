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
from MD_machineLearningFunctions import deNormalise

import seaborn as sb
import matplotlib.pyplot as plt
sb.set(context='talk', style='whitegrid', palette='Set1')
plt.rcParams["font.family"] = "serif"

# read in the data using pandas
datadir = '/home/ubuntu/Documents/NMDinStars/ML_models/SM/'
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

# Normalization is not necessary because MD's data
# is already normalized
normData = data

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
model.save('IBand_SM.h5')

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

# compute the ML errors for Iband and Ierr calculations
IbandErr = Iband - predI_deNorm
IerrErr = Ierr - predErr_deNorm

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18, 8))
ax1.hist(IbandErr)
ax2.hist(IerrErr)
ax1.set_ylabel('N')
ax1.set_xlabel('Error on I-Band Regression')
ax2.set_xlabel('Error on I-Band Error Regression')

fig.savefig('regression_error_hist.jpeg', transparent=False,
            bbox_inches='tight')

np.save('Iband_error', IbandErr)
np.save('Ierr_error', IerrErr)
