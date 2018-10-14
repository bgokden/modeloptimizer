from veriservice import VeriClient

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
# pprint library is used to make the output look more pretty
from pprint import pprint
import numpy as np
import veriservice as vs

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
# import pickle
import numpy as np
import gc
import time
import tempfile
import shutil
import sys
import gridfs


current_milli_time = lambda: int(round(time.time() * 1000))

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
test_url = "mongodb://automl:automl1234@ds111913.mlab.com:11913/modeldb"
db_name = "modeldb" # same as db name
# There will be two tables, project, models
projects_table_name = "project1" # will keep configuration for a project
models_table_name = "models" # will be used for table
data_host = "localhost:10000"
project_name = "sentencevectors"

client = MongoClient(test_url)
db=client[db_name]
models = db[models_table_name]
fs = gridfs.GridFS( db )

# project => {timestamp, project_name, settings, model_settings}
# model => { timestamp, project_name, loss, model_settings, modelfiles: {...} }
hpSeed = []
lowerLimit = []
upperLimit = []


def getOrCreateProject(project_name):
    projects = db[projects_table_name]
    project_conf = projects.find_one({'project_name': project_name})
    if project_conf is None:
        projects.insert_one({
            '_id' : project_name,
            'project_name': project_name,
            'timestamp': current_milli_time(),
            'settings': {
                'parallization': 1
            },
            'model_settings' : {
                'hyperparameter_initial_values': hpSeed,
                'hyperparameter_lower_limits': lowerLimit,
                'hyperparameter_upper_limits': upperLimit
            }
        })
        return getOrCreateProject(project_name)
    return project_conf

def getOrCreateBestTwoModels():
    models = db[models_table_name].find({'project_name': project_name}).sort("loss", pymongo.ASCENDING).limit(2)
    seed_models = []
    for model in models:
        pprint(model)
        seed_models.append(model)
    if len(seed_models) < 2:
        print('models are empty')
    return models


def saveKerasModel(project_name, loss, hyperparameters, keras_model):
    files = kerasModelToFileIds(keras_model)
    model_object = {
        'project_name': project_name,
        'loss': loss,
        'model_settings': {
            'hyperparameters': hyperparameters
        },
        'files': kerasModelToFileIds(keras_model)
    }
    pprint(model_object)
    saveModel(model_object)

def kerasModelToFileIds(keras_model):
    dirpath = tempfile.mkdtemp()
    path = dirpath+"/modelfile.h5"
    keras_model.save(path)
    fileIds = []
    fileID = fs.put( open( path, 'rb')  )
    fileIds.append(fileID)
    shutil.rmtree(dirpath)
    return fileIds

def readKerasModel(files):
    # keras model has only one file
    out = fs.get(files[0])
    dirpath = tempfile.mkdtemp()
    path = dirpath+"/modelfile.h5"
    model_file = open(path,'w')
    model_file.write(out)
    model_file.close()
    model = load_model(path)
    shutil.rmtree(dirpath)
    return model

def saveModel(model_object):
    db[models_table_name].insert_one(model_object)


# get best 2 models if not inilize random model configuration
# merge them
# create variations
# new conf is 2 best, 2 merged, 2 variations
# download the sample data from a veri instance
# train model with all of them
# upload the best one

def getData(host):
    vc = vs.VeriClient(host)
    data = vc.getLocalData()
    data_counter = 0
    features = []
    labels = []
    for datum in data:
        features.append(datum.feature)
        labels.append(datum.label)
        # print(len(datum.feature), datum.label)
        data_counter += 1
        if data_counter > 1000:
            break
    print("data size:",data_counter)
    return np.array(features),np.array(labels)


# generate lists of random integers and their sum
def getTrainData(features, labels, indexes, hsplit):
    idx = indexes[:hsplit]
    print(np.shape(labels))
    first = array(features)[idx].reshape((len(idx), 50, 300))
    second = array(labels)[idx].reshape((len(idx), 1, 1))
    return first, second


def getTestData(features, labels, indexes, hsplit):
    idx = indexes[hsplit:]
    first = array(features)[idx].reshape((len(idx), 50, 300))
    second = array(labels)[idx].reshape((len(idx), 1, 1))
    return first, second


# builds a model with given configuration array
# how to use a configuration array should be decided here
def getModel(hp):
    latent_dim = 300
    timesteps = 50
    input_dim = 300
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    sequence_autoencoder.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return sequence_autoencoder


def normalizeHp(hp, lowerL, upperL):
    for k in range(len(hp)):
        if hp[k] > upperL[k]:
            hp[k] = upperL[k]
        elif hp[k] < lowerL[k]:
            hp[k] = lowerL[k]
    return hp


def generateModels(project_conf):
    models_to_train = []
    seed_models = getOrCreateBestTwoModels()
    hps = []
    loss_scores = []
    for model in seed_models:
        models_to_train.append({
            'hyperparameters': model['model_settings']['hyperparameters'],
            model: readKerasModel(model.files)
        })
        hps.append(model['model_settings']['hyperparameters'])
        loss_scores.append(model['loss'])
    while len(hps) < 2:
        hps.append(project_conf['model_settings']['hyperparameter_initial_values'])
        loss_scores.append(sys.float_info.max) # this is not important just add a value
    generated_hps = getHyperparameters(hps, loss_scores)
    for j in range(len(generated_hps)):
        models_to_train.append({
            'hyperparameters': generated_hps[j],
            'model': getModel(generated_hps[j])
        })
    return models_to_train

def getHyperparameters(seeds, loss_scores):
    sorted_scores = sorted([*enumerate(loss_scores)], key=lambda x: x[1])
    index1 = sorted_scores[0][0]
    index2 = sorted_scores[1][0]
    first = seeds[index1]
    second = seeds[index2]
    if len(first) == 0 or len(second) == 0:
        return [[]]
    hps = []
    hsplit = int(np.ceil(len(first)*0.5))
    rindex = np.random.choice(len(first), len(first), replace=False)[:hsplit].astype(int)
    hp0 = first
    for i in rindex:
        hp0[i] = second[i]
    hp1 = second
    for i in rindex:
        hp1[i] = first[i]
    hps.append(hp0)
    hps.append(hp1)
    for a in range(2):
        hpa = np.multiply(hp0, (0.4 * np.random.rand(len(hp0))) + 0.8)
        hps.append(normalizeHp(hpa, lowerLimit, upperLimit))
    return hps

def trainCurrentModels(project_conf):
    features, labels = getData(data_host)
    len_features = len(features)
    hsplit = int(np.ceil(len_features*0.7))
    indexes = np.random.choice(len_features, len_features, replace=False)
    data_counter = 0
    for feature in features:
        print(len(feature), labels[data_counter])
        data_counter += 1
    # seed_models = getOrCreateBestTwoModels()
    # hps = initial_seeds
    # loss_scores = initial_loss_scores
    # loss_scores = []
    # best = []
    # best_loss_score = sys.float_info.max
    models_to_train = generateModels(project_conf)
    for model_to_train in models_to_train:
        model =  model_to_train['model']
        X, y = getTrainData(features, labels, indexes, hsplit)
        es = EarlyStopping(monitor='val_loss', min_delta=0.00001)
        model.fit(X, X, epochs=10, batch_size=256, validation_split=0.33, callbacks=[es])
        # evaluate Model
        X, y = getTestData(features, labels, indexes, hsplit)
        loss, acc = model.evaluate(X, X, verbose=0)
        # loss_scores.append(loss)
        model_to_train['loss'] = loss
        saveKerasModel(project_conf['project_name'],loss,model_to_train['hyperparameters'], model)
        gc.collect() # this may not be super meaningful



def main():
    project_conf = getOrCreateProject(project_name)
    print(project_conf)
    trial_number = 0
    trial_limit = 10
    while trial_number < trial_limit:
        trial_number += 1
        print("Running trial", trial_number)
        trainCurrentModels(project_conf)


if __name__ == "__main__":
    main()
