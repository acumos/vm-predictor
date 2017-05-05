from __future__ import print_function

import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator



def train_and_test(train_path, test_path, target_col, cols):
    print (">> Building model for target ", target_col)
    
    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print (">>   importing:", train_path)
    train_frame = h2o.import_file(path=train_path)    
    print (">>   importing:", test_path)
    test_frame = h2o.import_file(path=test_path)    
    
    print (">>   training...")
    #cols = [u'month', u'day', u'weekday', u'hour', u'minute']    
    res = rf_model.train (x=cols, y=target_col, training_frame=train_frame)
    
    print (">>   predicting...")
    preds = rf_model.predict(test_frame)

    # predictions are in preds
    print (">>   calculating AAE...")
    aae = calc_AAE (preds, test_frame, target_col)
    print (">>   AAE=", aae)
    
    predicted = preds.as_data_frame()    
    actual = test_frame.as_data_frame()
    
    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)

    return predicted, actual, aae


    
    
def OLD_predict_sliding_window (df, date_col='DATETIMEUTC', target_col):        # presumed that the incoming dataframe contains ONLY rows for the VM/subscriber/whatever
    training_period = 30 days
    predict_period = 7 days

    
    df = add_datetimes (df, date_col, "DT")     # all subsequent date calculation will use this column  (OR:  pre-process original)
    df = sort_by_time (df, "DT")
    
    train_start = 0
    
    while train_start < len(df):
        train_stop = walk_forward (df, train_start, training_period, "DT")
        predict_start = train_stop
        predict_stop = walk_forward (df, predict_start, predict_period, "DT")
        
        if predict_stop - predict_start < 1:
            break
            
        f_train = extract_file (df, train_start, train_stop)
        f_test = extract_file (df, predict_start, predict_stop)
        
        pred,act,err = train_and_test(f_train, f_test, target_col, feat_cols)
        remove_files ([f_train, f_test])
    
        train_start = walk_forward (df, train_start, predict_period, "DT")


        
def predict_sliding_window (df, date_col, target_col, feat_cols, train_inteval, predict_interval ):        # presumed that the incoming dataframe contains ONLY rows of interest
    DT = 'DT'
    df[DT] = pd.to_datetime(df[date_col])
    df = df.sort_values(DT)
    train_start = df[DT].iloc[0]        # OR:  min(df[date_col])
    
    while True:
        train_stop = train_start + train_interval
        predict_start = train_stop
        predict_stop = predict_start + predict_interval
        
        df_train = df[(df[DT] >= train_start) & (df[DT] < train_stop)]
        if len(df_train) < min_train_rows:
            break
            
        df_test = df[(df[DT] >= predict_start) & (df[DT] < predict_stop)]
        if len(df_test) < min_predict_rows:
            break
            
        df_train.to_csv(training_file_name, index=False)
        df_test.to_csv(testing_file_name, index=False)
        
        pred,act,err = train_and_test(training_file_name, testing_file_name, target_col, feat_cols)
        remove_files ([training_file_name, testing_file_name])
        
        train_start += predict_interval
        

        
# time_idx = pd.DatetimeIndex(df['DATETIMEUTC'])        # could be useful!
        
        
def example (filename, date_col='DATETIMEUTC', target_col='cpu_usage', features=[u'month', u'day', u'weekday', u'hour', u'minute'],
             training_interval_in_days=31.0, predict_interval_in_days=1.0):
          
    df = pd.read_csv(filename)
    trn_int = pd.Timedelta(days=training_inteval_in_days)
    prd_int = pd.Timedelta(days=predict_inteval_in_days)
    
    predict_sliding_window (df, date_col, target_col, features, trn_int, prd_int)
    
    
    
    



