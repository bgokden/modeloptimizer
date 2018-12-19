from trainer import Trainer
import os

import gensim.downloader as api
import spacy
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8
import numpy as np
import veriservice as vs
import time
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

nlp = spacy.load('en_core_web_sm')

word_model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
db_url = os.getenv('DB_URL', "")
db_name = os.getenv('DB_NAME', "modeldb") # same as db name
# There will be one table (collection for mongodb) models
models_table_name = os.getenv('DB_MODELS', "models") # will be used for table
data_host = os.getenv('DB_DATA_HOST', "localhost:10000")
project_name = os.getenv('PROJECT_NAME', "sentencevectors-10-2")
feature_dimention = os.getenv('FEATURE_DIMENTION', 100)
sequence_length = os.getenv('SEQUENCE_LENGTH', 10)

def text_2_features(text):
    features_array = []
    error_count = 0
    doc = nlp(text)
    for sentence in doc.sents:
        words = nlp(sentence.text)
        # print(len(words))
        features = np.array([])
        for token in words:
            try:
                # print( model.wv[token.text.lower()] )
                features = np.append(features, word_model.wv[token.text.lower().strip()])
                # break
            except:
                error_count+=1
        features_array.append(features)
    return np.array(features_array)

def predict(model, features):
    features = np.pad(features, ((0,0),(0,(feature_dimention*sequence_length)-np.shape(features)[1])), 'constant', constant_values=(0))
    features = features.reshape((len(features), sequence_length, feature_dimention))
    prediction = model.predict(features)[0]
    word = word_model.wv.similar_by_vector(prediction, topn=1)[0][0]
    return word
    target = list()
    for vector in prediction:
        word = word_model.wv.similar_by_vector(vector, topn=1)[0][0]
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

trainer = Trainer(project_name = project_name,
        db_url = db_url,
        db_name = db_name,
        models_table_name = models_table_name)


while True:
    model_object = trainer.get_best_model()
    model = model_object['models']['encoder']
    f_arr = text_2_features("he is very young.")
    # print("f_arr:", np.shape(f_arr[0]))
    print("predicted: ", predict(model, f_arr))
    time.sleep(10)
