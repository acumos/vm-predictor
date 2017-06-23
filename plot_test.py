import os
import pandas
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt


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
    xx = range(len(actual))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx,actual['usage'], 'b-', label='Actual')
    ax.plot(xx,predicted['predict'],'r-', label='Predicted')
    legend = ax.legend(loc='upper center', shadow=True)
    plt.title (file_path)
    #plt.show()
    fig.savefig(file_path+".png")
    #import pdb; pdb.set_trace()
    print "  done!"


if __name__ == "__main__":
    train_random_forest("ARCHER_WELL_8310006436036.csv", "usage")
    