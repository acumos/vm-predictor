# -*- coding: utf-8 -*-
# ================================================================================
# ACUMOS
# ================================================================================
# Copyright © 2017 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
# ================================================================================
# This Acumos software file is distributed by AT&T and Tech Mahindra
# under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================================================================

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
    print ">>   wrote: ", png_filename



def create_filename(png_dir, model_name, subscriber, vmachine):
    output_path = png_dir + "/" + model_name
    try:
        os.makedirs (output_path)
    except OSError:
        pass
    return output_path + "/" + subscriber + "-" + vmachine + ".png"



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



def process_vmachine(df, vmachine, target, png_base_path):
    #import pdb; pdb.set_trace()
    if not check_valid_target (df, target):
        print ">> ERROR -- target values are identical!"
    else:
        subscriber = get_subscriber(df)
        print ">> Rows:", df.shape[0]
        df1 = df[df['DATETIMEUTC'] < "2016-12-13 00:00:00"]
        if not check_valid_target (df1, target):
            print ">> ERROR -- training target values are identical!"
        else:
            df1.to_csv(vmachine + "_train.csv", index=False)
            df2 = df[df['DATETIMEUTC'] >= "2016-12-13 00:00:00"]
            df2.to_csv(vmachine + "_test.csv", index=False)
            predictions,actual,aae = train_and_test(vmachine + "_train.csv", vmachine + "_test.csv", target)
            title = target + "\n" + "subscriber %s" % subscriber + "\n" + vmachine + " (AAE=%s)" % aae
            fname = create_filename (png_base_path, target, subscriber, vmachine)
            draw_chart (title, predictions, actual, target, fname)
            print ">>   done!"



if __name__ == "__main__":
    import sys

    filename = sys.argv[1]
    target = sys.argv[2]
    topN = 40
    png_path = "./vm_models2"

    # load the dataframe & segregate it by vmachine
    print ">> processing file: ", filename
    df = pd.read_csv(filename)
    df.sort_values(by='DATETIMEUTC', inplace=True)
    vc = df['VM_ID'].value_counts()

    # build models for the top N vmachines
    for vm in vc.index[:topN+1]:                 # note: skip vmachine 'Unknown'
        print ">> VM: ", vm
        # save the vmachine rows as a file
        #fname = vmachine + ".csv"
        df2 = df[df['VM_ID']==vm]
        process_vmachine (df2, vm, target, png_path)


