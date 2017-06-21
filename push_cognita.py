
from __future__ import print_function

import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor        

import requests

#-----------------------------

import asimov_client as asimov
from asimov_client.api import Api



#-----------------------------


from crome import CromeProcessor



def SK_train (train_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(df_train[feat_cols], df_train[target_col])
    return rf
    
    #df_predict = pd.read_csv(test_path)
    #predicted = rf.predict(df_predict[feat_cols])
    #return predicted



def OLD_push_to_cognita (model, datafilename, feat_cols):
    username = 'mt531r'
    api = Api()
    if not api.users.exists({'username': username}):
        api.users.create(json={'username': username})
    
    template_name = 'crome-classifier-4'    
    
    # push the model to ASIMoV. this will take ~ 1m to build the solution (docker image)
    df = pd.read_csv(datafilename)
    df = df[feat_cols]
    asimov.push_model(model, df, username=username, template_name=template_name, create_template=True)
    return {'user': username, 'template': template_name}


def push_to_cognita (model, dataframe, feat_cols):
    username = 'mt531r'
    api = Api()
    if not api.users.exists({'username': username}):
        api.users.create(json={'username': username})
    
    template_name = 'crome-classifier-5'    
    
    # push the model to ASIMoV. this will take ~ 1m to build the solution (docker image)
    asimov.push_model(model, dataframe[feat_cols], username=username, template_name=template_name, create_template=True)
    return {'user': username, 'template': template_name}
    
    
    
    
def get_predictor (info):
    api = Api()
    user_id = api.users.one({'username': info['user']})['id']
    template_id = api.templates.one({'name': info['template'], 'owner': user_id})['id']
    model_id = 'latest'

    resp = requests.post('http://eve.asimov.research.att.com/solutions/running', json={'user': user_id,
                                                                                       'template': template_id,
                                                                                       'model': model_id}).json()
    ui = resp['ui']
    address = resp['address']
    predictor_api = "http://{}/predictor".format(address)
    
    print("Microservice running at {}".format(ui))
    return predictor_api
    
    
    
if __name__ == "__main__":
    import sys
    training_filename = sys.argv[1]
    
    target = "cpu_usage"
    features = ['day', 'weekday', 'hour', 'minute', 'hist-1D8H', 'hist-1D4H', 'hist-1D2H', 'hist-1D1H', 'hist-1D', 'hist-1D15m', 'hist-1D30m', 'hist-1D45m']

    print ("train: ", training_filename)
    cp = CromeProcessor (target, feats=features)
    df = cp.get_training_data (training_filename)
    model = cp.model.train(df, target, features)
    
    print ("push to cognita")
    info = push_to_cognita (model, df, features)

    print ("wait for microservice")
    time.sleep(5)              # give the microservice some time to load
    
    print ("get predictor")
    prediction_api = get_predictor(info)

    print ("prepare data")
    time.sleep(10)                          # needed to add this delay to avoid connection errors!!
    df = df[features]
    df = df.astype(int)
    lol = df.values.tolist()
    
    print ("do predictions")
    resp = requests.post(prediction_api, json=lol)
    preds = resp.json()
    import pdb; pdb.set_trace()

    print ("done!")
    
    
    
    #import pickle
    #pickle.dump(model, open("model.pkl", "wb"))
    
    
    