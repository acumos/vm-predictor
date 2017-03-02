import os
import pandas
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

import matplotlib.pyplot as plt
import datetime

def train_random_forest(file_path, target_col):
    print "Building model..."
    
    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print "  importing", file_path
    mainframe = h2o.import_file(path=file_path)    
    train_frame, test_frame = mainframe.split_frame([0.50])
    
    cols = [u'SUBSCRIBER_NAME', u'month', u'day', u'weekday', u'hour', u'minute']
    print "  training..."
    res = rf_model.train (x=cols, y=target_col, training_frame=train_frame)
    
    print "  predicting..."
    preds = rf_model.predict(test_frame)
    
    predicted = preds.as_data_frame()    
    actual = test_frame.as_data_frame()
    #xx = range(len(actual))
    #import pdb; pdb.set_trace()
    xx = actual['DATETIMEUTC'] / 1000               # remove trailing 000
    dates = xx.apply(lambda j:datetime.datetime.fromtimestamp(j))
    plt.plot_date(dates,actual['usage'], 'b-', label='Actual')
    plt.plot_date(dates,predicted['predict'],'r-', label='Predicted')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title (file_path)
    plt.show()
    import pdb; pdb.set_trace()
    print "  done!"


if __name__ == "__main__":
    train_random_forest("ARCHER_WELL_8310006436036.csv", "usage")
    