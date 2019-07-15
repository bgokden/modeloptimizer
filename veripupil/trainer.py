from veriservice import VeriClient

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
# pprint library is used to make the output look more pretty
from pprint import pprint
import numpy as np
import veriservice as vs
# import pickle
import numpy as np
import gc
import time
import tempfile
import shutil
import sys
import gridfs
import os
import jsonpickle

current_milli_time = lambda: int(round(time.time() * 1000))

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

import os
from veripupil import HyperparameterSet


class Trainer:

    def __init__(self,
                project_name = None,
                model_function = None,
                db_url = None,
                db_name = None,
                models_table_name = None,
                data_host = None,
                transform_function = None,
                seq2seq = False,
                feature_dimension_one = None,
                sequence_length_one = None,
                feature_dimension_two = None,
                sequence_length_two = None):
        self.project_name = project_name
        self.model_function = model_function
        self.transform_function = transform_function
        self.db_url = db_url
        self.db_name = db_name
        self.data_host = data_host
        self.models_table_name = models_table_name
        self.seq2seq = seq2seq
        self.feature_dimension_one = feature_dimension_one
        self.sequence_length_one = sequence_length_one
        self.feature_dimension_two = feature_dimension_two
        self.sequence_length_two = sequence_length_two
        self.feature_length = (self.feature_dimension_one  * self.sequence_length_one) + (self.feature_dimension_two * self.sequence_length_two)
        print("seq2seq:", self.seq2seq)
        print("Dimensions: ", self.feature_dimension_one, self.sequence_length_one, self.feature_dimension_two, self.sequence_length_two )

    def kerasModelToFileIds(self, model):
        fileIds = {}
        dirpath = tempfile.mkdtemp()
        client = MongoClient(self.db_url)
        db = client[self.db_name]
        fs = gridfs.GridFS( db )
        for file_name in model['models']:
            path = dirpath+"/"+file_name
            model['models'][file_name].save(path)
            print("model file size:", os.path.getsize(path))
            fileID = fs.put( open( path, 'rb'), project=model['project_name'], score=model['score'] )
            fileIds[file_name] = fileID
        shutil.rmtree(dirpath)
        return fileIds

    def saveModel(self, model):
        client = MongoClient(self.db_url)
        db = client[self.db_name]
        files = self.kerasModelToFileIds(model)
        hp_json = jsonpickle.encode(model['hyperparameters'])
        model['hyperparameters'] = hp_json
        model['models'] = {}
        model['files'] = files
        if '_id' in model:
            del model['_id']
        db[self.models_table_name].insert_one(model)

    def get_best_model(self):
        client = MongoClient(self.db_url)
        db=client[self.db_name]
        fs = gridfs.GridFS( db )
        models = db[self.models_table_name].find({'project_name': self.project_name}).sort("score", pymongo.ASCENDING).limit(1)
        for best_model in models:
            best_model['models'] = self.readKerasModels(best_model['files'])
            return best_model
        return None

    def get_latest_default_predictor(self):
        model_object = self.get_best_model()
        model = model_object['models'][model_object['model_to_optimize']]
        return model

    def getBestModelsAndCleanTheRest(self, limit = 2):
        client = MongoClient(self.db_url)
        db=client[self.db_name]
        fs = gridfs.GridFS( db )
        models = db[self.models_table_name].find({'project_name': self.project_name}).sort("score", pymongo.ASCENDING).limit(limit)
        seed_models = []
        score_max = 0
        for model in models:
            pprint(model)
            seed_models.append(model)
            if model['score'] > score_max:
                score_max = model['score']
        if len(seed_models) < limit:
            print('models are empty')
        else:
            result = db[self.models_table_name].delete_many({'score': {'$gt': score_max}})
            print(result.deleted_count, " documents deleted.")
            for model_file in fs.find({'project': self.project_name, 'score': {'$gt': score_max}},no_cursor_timeout=True):
                try:
                    fs.delete(model_file._id)
                except:
                    print('file deletion failed')
        return seed_models

    def delete_model(self, model):
        print("Deleting model with id", model['_id'])
        client = MongoClient(self.db_url)
        db = client[self.db_name]
        fs = gridfs.GridFS( db )
        model_object = db[self.models_table_name].find({'_id': model['_id']})
        db[self.models_table_name].delete_one({'_id': model['_id']})
        for file_name in model_object['files']:
            fs.delete(model_object['files'][file_name])


    def readKerasModels(self, files):
        # keras model has only one file
        client = MongoClient(self.db_url)
        db = client[self.db_name]
        fs = gridfs.GridFS( db )
        models = {}
        dirpath = tempfile.mkdtemp()
        for file_name in files:
            out = fs.get(files[file_name])
            path = dirpath+"/"+file_name
            model_file = open(path,'wb')
            model_file.write(out.read())
            model_file.close()
            models[file_name] = load_model(path)
        shutil.rmtree(dirpath)
        return models

    def generateModels(self, n = 6):
        models_to_train = []
        seed_models = self.getBestModelsAndCleanTheRest()
        hps_array = []
        for model in seed_models:
            hps_object = jsonpickle.decode(model['hyperparameters'])
            model['hyperparameters'] = hps_object
            model['models'] = self.readKerasModels(model['files'])
            models_to_train.append(model)
            hps_array.append(hps_object)
        if len(hps_array) > 2:
            child1, child2 = hps_array[0].cross_merge(hps_array[1])
            child1_model = self.model_function(child1)
            child1_model['score'] = sys.float_info.max
            models_to_train.append(child1_model)
            child2_model = self.model_function(child2)
            child2_model['score'] = sys.float_info.max
            models_to_train.append(child2_model)
        parent_hp = HyperparameterSet()
        if len(hps_array) > 1:
            parent_hp = hps_array[0]
        else:
            parent_hp = self.model_function()['hyperparameters']
        while len(models_to_train) < n:
            model = self.model_function(parent_hp.randomized())
            model['score'] = sys.float_info.max
            models_to_train.append(model)
        for model in models_to_train:
            model['project_name'] = self.project_name
        return models_to_train

    def getData(self, host):
        vc = vs.VeriClient(host)
        data = vc.getLocalData()
        data_counter = 0
        features = []
        labels = []
        for datum in data:
            features.append(datum.feature[:(self.feature_length)])
            labels.append(datum.label)
            # print(len(datum.feature), datum.label)
            data_counter += 1
            #if data_counter > 1000:
            #    break
        print("data size:",data_counter)
        return np.array(features),np.array(labels)

    def getTrainData(self, features, labels, indexes, hsplit):
        idx = indexes[:hsplit]
        if self.seq2seq :
            length1 = self.sequence_length_one*self.feature_dimension_one
            length2 = self.sequence_length_two*self.feature_dimension_two
            features = np.pad(features, ((0,0),(0,self.feature_length-np.shape(features)[1])), 'constant', constant_values=(0))
            features1 = features[:,:length1]
            first = array(features1)[idx].reshape((len(idx), self.sequence_length_one, self.feature_dimension_one))
            features2 = features[:,-length2:]
            second = array(features2)[idx].reshape((len(idx), self.sequence_length_two, self.feature_dimension_two))
        else:
            features = np.pad(features, ((0,0),(0,self.feature_length-np.shape(features)[1])), 'constant', constant_values=(0))
            first = array(features)[idx].reshape((len(idx), self.sequence_length_one, self.feature_dimension_one))
            second = array(labels)[idx].reshape((len(idx), 1, 1))
        return self.transform_function(first, second)


    def getTestData(self, features, labels, indexes, hsplit):
        idx = indexes[hsplit:]
        if self.seq2seq :
            length1 = self.sequence_length_one*self.feature_dimension_one
            length2 = self.sequence_length_two*self.feature_dimension_two
            features = np.pad(features, ((0,0),(0,self.feature_length-np.shape(features)[1])), 'constant', constant_values=(0))
            features1 = features[:,:length1]
            first = array(features1)[idx].reshape((len(idx), self.sequence_length_one, self.feature_dimension_one))
            features2 = features[:,-length2:]
            second = array(features2)[idx].reshape((len(idx), self.sequence_length_two, self.feature_dimension_two))
        else:
            features = np.pad(features, ((0,0),(0,self.feature_length-np.shape(features)[1])), 'constant', constant_values=(0))
            first = array(features)[idx].reshape((len(idx), self.sequence_length_one, self.feature_dimension_one))
            second = array(labels)[idx].reshape((len(idx), 1, 1))
        return self.transform_function(first, second)

    def iterative_learning_function(self, X, y, iteration):
        Xi = np.copy(X)
        yi = np.copy(y)
        Xi[:, iteration:, :] = 0
        yi[:, iteration:, :] = 0
        return Xi, yi

    def trainCurrentModels(self, trial):
        features, labels = self.getData(self.data_host)
        len_features = len(features)
        hsplit = int(np.ceil(len_features*0.8))
        indexes = np.random.choice(len_features, len_features, replace=False)
        models_to_train = self.generateModels()
        model_iteration = 0
        improved_models = []
        for model_to_train in models_to_train:
            try:
                print("Running model ", model_iteration, "with initial loss", model_to_train['score'])
                model =  model_to_train['models'][model_to_train['model_to_optimize']]
                hps = model_to_train['hyperparameters']
                X, y = self.getTrainData(features, labels, indexes, hsplit)
                loss_min_delta = hps.get(name = 'loss_min_delta',type='loss_min_delta', hint=0.000001)
                print(model.summary())
                batch_size = hps.get(name='batch_size', type='batch_size', hint=32, maximum=256)
                hps.print_all()
                # iteration = 1
                # while iteration < self.sequence_length:
                #    Xi, yi = self.iterative_learning_function(X, y, iteration)
                #    print("iteration:", iteration, "for trial:", trial)
                #    es_temp = EarlyStopping(monitor='val_loss', min_delta=loss_min_delta, patience=2, restore_best_weights=True)
                #    model.fit(Xi, yi, epochs=100, batch_size=batch_size, validation_split=0.33, callbacks=[es_temp])
                #    iteration = iteration * 2 # 1, 2 , 4, 8, 16, 32 ... 50
                # do a last training with everything
                es = EarlyStopping(monitor='val_loss', min_delta=loss_min_delta, patience=5, restore_best_weights=True)
                model.fit(X, y, epochs=100, batch_size=batch_size, validation_split=0.33, callbacks=[es])
                # evaluate Model
                X, y = self.getTestData(features, labels, indexes, hsplit)
                loss, acc = model.evaluate(X, y, verbose=0)
                print("New evaluated loss:", loss, " acc:", acc)
                # loss_scores.append(loss)
                if model_to_train['score'] > loss:
                    model_to_train['score'] = loss
                    model_to_train['models'][model_to_train['model_to_optimize']] = model
                    improved_models.append(model_to_train)
            except Exception as e:
                # Exceptions can happen sometimes some architectures are not compatible
                print("Exception while training", e)
            model_iteration+=1
        if len(improved_models) > 0:
            best_model = improved_models[0]
            for improved_model in improved_models:
                if best_model['score'] > improved_model['score']:
                    best_model=improved_model
            self.saveModel(best_model)
            if '_id' not in best_model:
                # this is a critical operation so it would be better to ..
                # check if data size is large enough or new score is close to old one
                for model_to_delete in models_to_train:
                    if '_id' in model_to_delete:
                        # an old best model lost its place due to paradigm shift
                        this.delete_model(model_to_delete)
        else:
            print("Models are not improved")


    def run(self):
        trial_number = 0
        trial_limit = 10
        while trial_number < trial_limit:
            trial_number += 1
            print("Running trial", trial_number)
            self.trainCurrentModels(trial_number)
            gc.collect() # this may not be super meaningful
