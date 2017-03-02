import os
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter

    
# derived from calc_AAE.py    
def calc_AAE (predict_h2o, actual_h2o, target_col):           
    # we must convert to pandas first to preserve NAN entries
    actual = actual_h2o.as_data_frame()
    actual = actual[target_col]
    predicted = predict_h2o.as_data_frame()
    predicted = predicted['predict']

    numerator = abs(predicted - actual)
    denominator = predicted.combine(actual, max)
    aae = numerator / denominator
    return aae.mean()
    

    
def train_and_test(train_path, test_path, target_col):
    print ">> Building model for target ", target_col
    
    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print ">>   importing:", train_path
    train_frame = h2o.import_file(path=train_path)    
    print ">>   importing:", test_path
    test_frame = h2o.import_file(path=test_path)    
    
    cols = [u'month', u'day', u'weekday', u'hour', u'minute']
    print ">>   training..."
    res = rf_model.train (x=cols, y=target_col, training_frame=train_frame)
    
    print ">>   predicting..."
    preds = rf_model.predict(test_frame)

    # predictions are in preds
    print ">>   calculating AAE..."
    aae = calc_AAE (preds, test_frame, target_col)
    print ">>   AAE=", aae
    
    predicted = preds.as_data_frame()    
    actual = test_frame.as_data_frame()
    
    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)

    return predicted, actual, aae


def draw_chart (chart_title, predicted, actual, target_col, png_filename):
    xx = actual['DATETIMEUTC'] / 1000               # remove trailing 000s
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    dates = xx.apply(lambda j:datetime.datetime.fromtimestamp(j))
    ordinals = [matplotlib.dates.date2num(d) for d in dates] 
    
    ax.plot_date(ordinals,actual[target_col], 'b-', label='Actual')
    ax.plot_date(ordinals,predicted['predict'],'r-', label='Predicted')

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
    print ">>   wrote: ", png_filename
    


def create_filename(png_dir, model_name, entity, subscriber=None):
    output_path = png_dir + "/" + model_name
    try:
        os.makedirs (output_path)
    except OSError:
        pass
    if subscriber:
        return output_path + "/" + subscriber + "-" + entity + ".png"
    else:
        return output_path + "/" + entity + ".png"



def check_valid_target (df, target):
    vc = df[target].value_counts()
    if len(vc) < 2:
        return False
    valid = 0
    for idx,val in vc.iteritems():
        if val > 1:
            valid += 1
            if valid > 1:
                return True
    return False
        


def get_subscriber (df):
    subname = ""
    subs = set(df['SUBSCRIBER_NAME'])
    for s in subs:
        subname = subname + s
    return subname
   
        
    
def process_entity(df, entity, target, png_base_path, subscriber):
    aae = None
    fname = create_filename (png_base_path, target, entity, subscriber)
    if os.path.exists(fname):
        print ">> ALREADY EXISTS!"
    elif not check_valid_target (df, target):
        print ">> ERROR -- target values are identical!"
    else:
        df1 = df[df['DATETIMEUTC'] < "2016-12-13 00:00:00"]
        df2 = df[df['DATETIMEUTC'] >= "2016-12-13 00:00:00"]
        if not check_valid_target (df1, target) or not check_valid_target (df2, target):
            print ">> ERROR in train or test data!"
        else:
            df1.to_csv(entity + "_train.csv", index=False)
            df2.to_csv(entity + "_test.csv", index=False)
            predictions,actual,aae = train_and_test(entity + "_train.csv", entity + "_test.csv", target)
            if subscriber:
                title = target + "\n" + "subscriber %s" % subscriber + "\n" + entity + " (AAE=%s)" % aae
            else:
                title = target + "\n" + entity + " (AAE=%s)" % aae
            draw_chart (title, predictions, actual, target, fname)
            print ">>   done!"
    return aae
            
'''            
def process_subscriber(df, subscriber, target, png_base_path):
    if len(df[target].value_counts()) < 2:
        print ">> ERROR -- all target values are identical!"
    else:
        print ">> Rows:", df.shape[0]
        df.to_csv(subscriber + ".csv", index=False)
        predictions,actual,aae = train_and_test(subscriber + ".csv", target)
        title = target + "\n" + subscriber + " (AAE=%s)" % aae
        fname = create_filename (png_base_path, target, subscriber)
        draw_chart (title, predictions, actual, target, fname)
        print ">>   done!"
'''            

def write_result (result_file, score, id_column, entity, target, subscriber, rows):
    if score==None:
        score = "N/A"
    if not os.path.exists(result_file):
        fp = open(result_file, "w")
        fp.write ("Date, Target, Subscriber, VM, AAE, Rows\n")
        fp.close()
    fp = open(result_file, "a")
    the_date = datetime.datetime.now()
    if subscriber:
        fp.write ("%s, %s, %s, %s, %s, %s\n" % (the_date, target, subscriber, entity, score, rows))
    else:
        fp.write ("%s, %s, %s, %s, %s, %s\n" % (the_date, target, entity, "all", score, rows))
    fp.close()

        
        
if __name__ == "__main__":        
    import sys

    topN = 3
    png_path = "./test_sm2"
    id_column = 'VM_ID'
    min_rows = 1000
    #id_column = 'SUBSCRIBER_NAME'
    
    filename = sys.argv[1]
    target = sys.argv[2]
    id_column = sys.argv[3]

    result_file = "summary_" + target + ".csv"
    
    # load the dataframe & segregate it by vmachine
    print ">> processing file: ", filename
    print ">> ID column: ", id_column
    df = pd.read_csv(filename)
    df.sort_values(by='DATETIMEUTC', inplace=True)
    vc = df[id_column].value_counts()

    # build models for the top N vmachines
    #for entity in vc.index[:topN+1]:                 # note: skip vmachine 'Unknown'
    #for entity in vc.index:
    print "!!!!!!!!!!! TEST VERSION !!!!!!!!!!!!"
    for entity in ['40470839-1a9d-49c9-9e94-4b13c537f8ab']:
        print ">> %s %s: " % (target, id_column), entity
        df2 = df[df[id_column]==entity]
        rows = df2.shape[0]
        print ">> Rows:", rows
        if rows < min_rows:
            break
        subscriber = get_subscriber(df2) if  id_column != 'SUBSCRIBER_NAME' else None
        score = process_entity (df2, entity, target, png_path, subscriber)
        if score != None:
            write_result (result_file, score, id_column, entity, target, subscriber, rows)
        

