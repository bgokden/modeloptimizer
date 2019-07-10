from veripupil.trainer import Trainer
import os

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, RepeatVector, Dropout
from keras.models import Model
from keras.models import load_model

from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from veripupil.hyperparameters import HyperparameterSet

def getModel(hps = HyperparameterSet()):
    model = Sequential()
    model.add(Dense(hps.get(type = 'dimension', hint = 512, maximum = 1024), input_shape=(784,)))
    a = hps.get(type = 'keras_activation', hint = 'relu')
    print(a)
    model.add(Activation(a))
    model.add(Dropout(hps.get(type = 'dropout', hint = 0.2)))
    iterator = hps.get(type='choice', hint = 1, choices = [1, 3, 5, 8])
    #hps.get(type='iterator', hint = 1, maximum = 5)
    for i in range(iterator):
        model.add(Dense(hps.get(scope='layers', type = 'dimension', hint = 512, maximum = 1024)))
        model.add(Activation(hps.get(scope='layers', type = 'keras_activation', hint = 'relu')))
        model.add(Dropout(hps.get(scope='layers', type = 'dropout', hint = 0.2)))
    model.add(Dense(10))
    model.add(Activation(hps.get(type = 'keras_activation', hint = 'softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=hps.get(type = 'keras_optimizer'), metrics=['accuracy'])

    model_object = {}
    model_object['type'] = 'keras'
    model_object['hyperparameters'] = hps
    model_object['model_to_optimize'] = 'model'
    model_object['models'] = {'model': model }
    return model_object

def transform(features, labels):
    features = features.reshape((len(features), 784))
    nb_classes = 10
    labels = labels.reshape((len(labels), 1))
    labels = np_utils.to_categorical(labels, nb_classes)
    return features, labels

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
db_url = os.getenv('DB_URL', "")
db_name = os.getenv('DB_NAME', "modeldb") # same as db name
# There will be one table (collection for mongodb) models
models_table_name = os.getenv('DB_MODELS', "models") # will be used for table
data_host = os.getenv('DB_DATA_HOST', "localhost:10000")
project_name = os.getenv('PROJECT_NAME', "mnist")
feature_dimension = os.getenv('FEATURE_DIMENSION', 784)
sequence_length = os.getenv('SEQUENCE_LENGTH', 1)

Trainer(project_name = project_name,
        model_function = getModel,
        db_url = db_url,
        db_name = db_name,
        models_table_name = models_table_name,
        data_host = data_host,
        transform_function = transform,
        feature_dimension = feature_dimension,
        sequence_length = sequence_length
        ).run()
