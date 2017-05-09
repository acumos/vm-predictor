from __future__ import print_function

import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

    


    
def H2O_train_and_predict(train_path, test_path, target_col, cols):
    print (">> Building model for target ", target_col)
    
    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print (">>   importing:", train_path)
    train_frame = h2o.import_file(path=train_path)    
    print (">>   importing:", test_path)
    test_frame = h2o.import_file(path=test_path)    
    
    print (">>   training...")
    res = rf_model.train (x=cols, y=target_col, training_frame=train_frame)
    
    print (">>   predicting...")
    preds = rf_model.predict(test_frame)

    predicted = preds.as_data_frame()    
    
    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)

    return predicted
    
    
    
   
        
def predict_sliding_windows (df, timestamp_col, target_col, feat_cols, train_interval, predict_interval ):        # presumed that the incoming dataframe contains ONLY rows of interest
    min_train_rows = 1000
    min_predict_rows = 1
    training_file_name = "./train.csv"
    testing_file_name = "./test.csv"
    DT = timestamp_col
    predict_col = 'predict'                 # H2O convention

    train_start = df[DT].iloc[0]
    result = pd.DataFrame(columns=[predict_col, target_col, DT])
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
        
        preds = H2O_train_and_predict(training_file_name, testing_file_name, target_col, feat_cols)
        kwargs = {predict_col : preds[predict_col].values}
        df_test = df_test.assign (**kwargs)
        result = pd.concat([result, df_test[[predict_col, target_col, DT]]])
        
        #remove_files ([training_file_name, testing_file_name])
        train_start += predict_interval
        
    return result

    
        
       
        
def train_test (filename, features, date_col, target_col, training_interval_in_days, predict_interval_in_days):
    df = pd.read_csv(filename)
    trn_int = pd.Timedelta(days=training_interval_in_days)
    prd_int = pd.Timedelta(days=predict_interval_in_days)
    
    DT = 'DT'
    df[DT] = pd.to_datetime(df[date_col])       # WHO does this conversion and WHEN ?
    df = df.sort_values(DT)

    df_result = predict_sliding_windows (df, DT, target_col, features, trn_int, prd_int)
    return df_result[DT], df_result['predict'], df_result[target_col]
    


# PANDAS CUTOFF LINE ---------------------------------------------------------------------------------------------------------

import os
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter
    
    
# derived from calc_AAE.py    
def calc_AAE (predictions, actual):               # input:  two pd.Series
    numerator = abs(predictions - actual)
    denominator = predictions.combine(actual, max)
    aae = numerator / denominator
    return aae.mean()
    
    

def draw_chart (chart_title, predicted, actual, dates, png_filename):       # three pd.Series
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ordinals = [matplotlib.dates.date2num(d) for d in dates] 
    
    ax.plot_date(ordinals,actual,'b-', label='Actual')
    ax.plot_date(ordinals,predicted,'r-', label='Predicted')

    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
    ax.xaxis.set_minor_locator(HourLocator())
    ax.autoscale_view()
    ax.grid(True)
    fig.autofmt_xdate()
    
    legend = ax.legend(loc='upper right', shadow=True)
    plt.title (chart_title)
    fig.savefig(png_filename)
    plt.close()
    print (">>   wrote: ", png_filename)
    

    
    


def compose_filename(png_dir, model_name, entity, subscriber=None):
    output_path = png_dir + "/" + model_name
    try:
        os.makedirs (output_path)
    except OSError:
        pass
    if subscriber:
        return output_path + "/" + subscriber + "-" + entity + ".png"
    else:
        return output_path + "/" + entity + ".png"

   
    
    

def process_crome_data_file (data_file_name, target, subscriber=None, train_size_days=31, predict_size_days=7, png_base_path="."):
    dates, predicted, actual = train_test (data_file_name, features=[u'month', u'day', u'weekday', u'hour', u'minute'], date_col='DATETIMEUTC',
                               target_col=target, training_interval_in_days=train_size_days, predict_interval_in_days=predict_size_days)
    aae = calc_AAE (predicted, actual)
    chart_file = compose_filename (png_base_path, target, data_file_name, subscriber)
    title = target + "\n" + data_file_name + " (AAE=%s)" % aae
    title += "\n" + "train=%d, test=%d" % (train_size_days, predict_size_days)
    import pdb; pdb.set_trace()
    draw_chart (title, predicted, actual, dates, chart_file)
    


if __name__ == "__main__":
    '''
    entity = "08afdbcc-55b2-404f-9c13-2af69cdcf611.csv"
    target = "cpu_usage"
    subscriber = None
    
    res = example (entity, target_col=target, training_interval_in_days=25.0, predict_interval_in_days=0.25)
    aae = calc_AAE (res['predict'], res[target])

    fname = compose_filename (png_base_path, target, entity, subscriber)

    
    title = target + "\n" + entity + " (AAE=%s)" % aae
    draw_chart (title, res['predict'], res['actual'], res['date'], fname)
    '''
    
    # note this file is only 1 month long
    process_crome_data_file ("08afdbcc-55b2-404f-9c13-2af69cdcf611.csv", "cpu_usage", train_size_days=15, predict_size_days=7)
    
    
    
    
    
    
    
