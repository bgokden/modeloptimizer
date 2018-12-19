from trainer import Trainer
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
from hyperparameters import HyperparameterSet

def getModel(hps = HyperparameterSet()):
    latent_dim = hps.get(name = 'latent_dim', type = 'dimension', hint = 100, maximum = 1024) #300
    timesteps = 10
    input_dim = 100
    dropout_hint = 0.01
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim, return_sequences=True)(inputs)
    encoded = Dropout(hps.get(type = 'dropout', hint = dropout_hint))(encoded)
    iterator = hps.get(name="encoder_iterator", type='iterator', hint = 1, maximum = 5)
    for i in range(iterator):
        encoded = LSTM(hps.get(scope='encoder_loop', type = 'dimension', hint = input_dim, maximum = 1024), return_sequences=True)(encoded)
        encoded = Dropout(hps.get(scope='encoder_loop', type = 'dropout', hint = dropout_hint))(encoded)
    encoded = LSTM(input_dim, return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    for i in range(hps.get(name="decoder_iterator", type='iterator', hint = 1, maximum = 5)):
        decoded = LSTM(hps.get(scope='decoder_loop',type = 'dimension', hint = input_dim, maximum = 1024), return_sequences=True)(decoded)
        decoded = Dropout(hps.get(scope='decoder_loop', type = 'dropout', hint = dropout_hint))(decoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    sequence_autoencoder.compile(loss='hinge', optimizer=hps.get(type = 'keras_optimizer'), metrics=['accuracy'])
    # print(sequence_autoencoder.summary())
    model = {}
    model['type'] = 'keras'
    model['hyperparameters'] = hps
    model['model_to_optimize'] = 'sequence_autoencoder'
    model['models'] = {'sequence_autoencoder': sequence_autoencoder, 'encoder': encoder }
    return model

def transform(features, labels):
    return features, features

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
db_url = os.getenv('DB_URL', "")
db_name = os.getenv('DB_NAME', "modeldb") # same as db name
# There will be one table (collection for mongodb) models
models_table_name = os.getenv('DB_MODELS', "models") # will be used for table
data_host = os.getenv('DB_DATA_HOST', "localhost:10000")
project_name = os.getenv('PROJECT_NAME', "sentencevectors-10-2")
feature_dimension = os.getenv('FEATURE_DIMENSION', 100)
sequence_length = os.getenv('SEQUENCE_LENGTH', 10)

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
