'''
ML classifier algorithm to differentiate between good mesa models
and models that are too old or HE shell flash
'''
# imports
import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from ML_Helpers import *

# read in   the data using pandas
datadir = '/home/nfranz/research/NMDinStars/analysis/output_analysis/'
infile = os.path.join(datadir, 'allData.csv')
allData = pd.read_csv(infile) # time is in minutes
allData = allData[allData.flag != 1] # cut out flag = 1
data = allData[['mass', 'y', 'z', 'mu', 'M_I', 'M_I_err', 'V_I', 'V_I_err']]
print(data[data['mu']==0])
# set hyper parameters
nLayers = 8
epochs = 120
batch = 120

# normalize the data
normData, minVal, maxVal = minNormalize(data)
# write constants to a file
minMax = pd.concat([minVal, maxVal], axis=1)
minMax.columns = ['min', 'max']
minMax.to_csv('norm_const.txt')

print(normData)
# remove the I-band and M_I columns
normData.drop(['M_I', 'M_I_err', 'V_I', 'V_I_err'], inplace=True, axis=1)

# add flag labels to the dataset for splitting
normData['flag'] = allData['flag']

# randomly split the data
train, val, test = splitData(normData)

# pop flags
trainFlags = train.pop('flag')
valFlags = val.pop('flag')
testFlags = test.pop('flag')

# resample training data
print('Resample the training data...\n')
smote = SMOTE(k_neighbors=10)
trianResample, trainFlagsResample = smote.fit_resample(train, trainFlags)

# implement one-hot encoding
trainFlags = pd.get_dummies(trainFlags, prefix='flag', prefix_sep='')
valFlags = pd.get_dummies(valFlags, prefix='flag', prefix_sep='')
#testFlags = pd.get_dummies(testFlags, prefix='flag', prefix_sep='')

# generate the ML model
model = buildModel(nLayers, nOutputs=3,
                   metrics=['categorical_accuracy',
                            'FalsePositives',
                            'FalseNegatives'],
                   loss='categorical_crossentropy',
                   optimizer=Adam(learning_rate=0.001,
                                  beta_1=0.1,
                                  beta_2=0.999))

# train the model
print('Training the model...\n')
cb = EarlyStopping(monitor='val_loss',
                   min_delta=0.001,
                   patience=5,
                   restore_best_weights=True)
lrCalib = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.01,
                            patience=3)
hist = model.fit(train, trainFlags,
                 validation_data=(val, valFlags),
                 epochs=epochs,
                 batch_size=batch,
                 verbose=0,
                 callbacks=[cb, lrCalib])

# save the model
model.save('classify_mesa.h5')

# plot accuracy and loss
plotLoss(hist.history)
plotAccuracy(hist.history, key='categorical_accuracy')
plotFalseNegPos(hist.history)


# test the model
pred  = model.predict(test)

outFlag = []
for val in np.argmax(pred, axis=1):
    if val > 0:
        outFlag.append(val+1)
    else:
        outFlag.append(val)

whereEq = np.where((testFlags == outFlag) == True)[0]
acc = len(whereEq) / len(testFlags)

plotCompareHist(testFlags, outFlag)

forFile = ''
forFile += f"Loss: {hist.history['val_loss'][-1]}\n"
forFile += f'Testing Accuracy: {acc}' 

print(forFile)

# save the model
model.save('classify_mesa.h5')

# save a file of this information
with open('classifier_metrics.txt', 'w') as f:
    f.write(forFile)
