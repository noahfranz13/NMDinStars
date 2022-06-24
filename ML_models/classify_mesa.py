'''
ML classifier algorithm to differentiate between good mesa models
and models that are too old or HE shell flash
'''
# imports
import os
import pandas as pd

#from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE

from ML_Helpers import *

# read in   the data using pandas
datadir = '/home/ubuntu/Documents/NMDinStars/ML_models/'
infile = os.path.join(datadir, 'fulldata.txt')
allData = pd.read_csv(infile) # time is in minutes
allData = allData[allData.flag != 1] # cut out flag = 1
data = allData[['mass', 'Y', 'Z']]

# set hyper parameters
nLayers = 8
epochs = 120
batch = 1200

# normalize the data
#normData, minVal, maxVal = minNormalize(data)
normData = data # just because the input is already normalized

normData['flag'] = allData['flag']

# randomly split the data
train, val, test = splitData(normData)

# pop flags
trainFlags = train.pop('flag')
valFlags = val.pop('flag')
testFlags = test.pop('flag')

# resample training data
print('Resample the training data...\n')
trianResample, trainFlagsResample = SMOTE().fit_resample(train, trainFlags)

# implement one-hot encoding
trainFlags = pd.get_dummies(trainFlags, prefix='flag', prefix_sep='')
valFlags = pd.get_dummies(valFlags, prefix='flag', prefix_sep='')
#testFlags = pd.get_dummies(testFlags, prefix='flag', prefix_sep='')

# generate the ML model
model = buildModel(nLayers, nOutputs=3, metrics=['categorical_accuracy'], loss='categorical_crossentropy')

# train the model
print('Training the model...\n')
hist = model.fit(train, trainFlags,
                 validation_data=(val, valFlags),
                 epochs=epochs,
                 batch_size=batch,
                 verbose=1)

plotLoss(hist.history)
plotAccuracy(hist.history, key='categorical_accuracy')

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

plotCompare(testFlags, outFlag)
plotCompareHist(testFlags, outFlag)


print('Testing Accuracy: ', acc)

model.save('classify_mesa')
