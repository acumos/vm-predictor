
from __future__ import print_function

import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor        

import requests

#-----------------------------

import cognita_client 
from cognita_client.api import Api


#-----------------------------


from crome import CromeProcessor


def push_to_cognita (model, dataframe, feat_cols, api=None):
    apiObj = Api()
    #username = 'mt531r'
    #if not api.users.exists({'username': username}):
    #    api.users.create(json={'username': username})
    
    template_name = 'vm_predictor'    
    
    # push the model to cognita. this will take ~ 1m to build the solution (docker image)
    cognita_client.push.push_sklearn_model(model, dataframe[feat_cols], 
                                            extra_deps=None, api=api)
    return {'template': template_name}
    
    
    
def get_predictor (info):
    api = Api()
    user_id = api.users.one({'username': info['user']})['id']
    template_id = api.templates.one({'name': info['template'], 'owner': user_id})['id']
    model_id = 'latest'

    resp = requests.post('http://eve.cognita.research.att.com/solutions/running', json={'user': user_id,
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
    tmpfile = "training.csv"
    
    print ("train: ", training_filename)
    cp = CromeProcessor (target, feats=features)
    model = cp.build_model_from_CSV (training_filename, datafile_out=tmpfile)
    df = pd.read_csv(tmpfile)
    
    print ("push to cognita")
    info = push_to_cognita (model, df, features, api="http://localhost:8887/v1/models")

    print ("wait for microservice")
    time.sleep(5)              # give the microservice some time to load
    
    print ("additional testing is deferred")
    """
    print ("get predictor")
    prediction_api = get_predictor(info)

    print ("prepare data")
    time.sleep(10)                          # needed to add this delay to avoid connection errors!!
    df = df[features]
    
    # Convert to list of lists.
    df['__tmp__'] = 'a'                     # A sneaky way to preserve types:  typecast won't be applied if an object (string) is present
    lol = df.values.tolist()
    lol = [n[:-1] for n in lol]             # remove the temp string before posting
    
    print ("do predictions")
    resp = requests.post(prediction_api, json=lol)
    preds = resp.json()
    
    print ("example predictions:", preds[:5])
    """

    print ("done!")
    
    
    
