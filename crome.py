from __future__ import print_function

import pandas as pd
import numpy as np
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter

    
def H2O_train_and_predict(train_path, test_path, target_col, feat_cols):
    print (">> Building model for target ", target_col)
    
    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print (">>   importing:", train_path)
    train_frame = h2o.import_file(path=train_path)    
    print (">>   importing:", test_path)
    test_frame = h2o.import_file(path=test_path)    
    
    print (">>   training...")
    res = rf_model.train (x=feat_cols, y=target_col, training_frame=train_frame)
    
    print (">>   predicting...")
    preds = rf_model.predict(test_frame)

    predicted = preds.as_data_frame()    
    
    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)

    return predicted
    

def get_busy_hour (arr, period):
    high = None
    best = None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x : x+period])
        if not high or score > high:
            best, high = x, score
    return arr.index[best]
          

def get_busy_avg (arr, period):
    high = None
    best = None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x : x+period])
        if not high or score > high:
            best, high = x, score
    return high


  
          
          
class CromeProcessor:
    # See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for resample strings
    def __init__ (self, target_col, date_col='DATETIMEUTC', train_size_days=31, predict_size_days=1, resample_str="15min", png_base_path="." ):
        self.target_col = target_col
        self.train_interval = pd.Timedelta(days=train_size_days)
        self.predict_interval = pd.Timedelta(days=predict_size_days)
        self.resample_str = resample_str
        self.png_base = png_base_path
        self.date_col=date_col
        self.min_train_rows = 300
        self.min_predict_rows = 1
        self.pctile = 95
        self.training_file_name = "./train.csv"
        self.testing_file_name = "./test.csv"
        self.busyHours = 4        # in hours
        self.STD = True
        self.VAR = True
        self.showRaw = True
               
    
    def process_file (self, filename):
        df = pd.read_csv(filename)
        print (">> %s: %s rows" % (filename, len(df)))
        df, features = self.transform_dataframe (df)
        df_result = self.predict_sliding_windows (df, features)
        output = {}
        if len(df_result) > 0:
            if self.showRaw:
                self.add_view (output, "original", df_result, False)
            if self.pctile:
                self.add_view (output, "percentile_%s" % self.pctile, df_result.resample("1D").apply(lambda x: np.percentile(x, self.pctile)), True)
            if self.STD:
                self.add_view (output, "std", df_result.resample("1D").apply(lambda x: np.std(x)), False)
            if self.VAR:
                self.add_view (output, "variance", df_result.resample("1D").apply(lambda x: np.var(x)), False)
            if self.busyHours:
                df_hour = df_result.resample("1H").mean()
                self.add_view (output, "busy_avg_%sH" % self.busyHours, df_hour.resample("1D").apply(lambda x: get_busy_avg(x, self.busyHours)), True)
        else:
            print (">> insufficient data")
        return output
            
    
    def transform_dataframe (self, df):       # could be a pipeline
        # 1. convert to datetime index
        DT = 'DT'
        df[DT] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(DT)
        df.index = pd.DatetimeIndex(df['DT'])    
        df.drop (self.date_col, axis=1, inplace=True)
        df.drop (DT, axis=1, inplace=True)
        
        # 2. re-sample at desired interval, e.g. "15min" or "1H".  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        if self.resample_str:
            df = df.resample (self.resample_str).mean()          # may use 'sum' here instead of 'mean' !!
        
        # 3. add time features
        df['month'] =  df.index.month
        df['day'] =  df.index.day
        df['weekday'] = df.index.weekday
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # 4. add any other features here (including any "pass-through" features)
        # ...
        
        # TO DO :  drop all columns which are not target or features  ??
        return df, [u'month', u'day', u'weekday', u'hour', u'minute']

    
    def predict_sliding_windows (self, df, feat_cols):        # presumed that the incoming dataframe contains ONLY rows of interest
        predict_col = 'predict'                 # H2O convention
        train_start = df.index[0]
        result = pd.DataFrame()
        while True:
            train_stop = train_start + self.train_interval
            predict_start = train_stop
            predict_stop = predict_start + self.predict_interval
            
            df_train = df[train_start:train_stop]
            if len(df_train) < self.min_train_rows:
                break
                
            df_test = df[predict_start:predict_stop]
            if len(df_test) < self.min_predict_rows:
                break
                
            print (">>   train/predict sizes: ", len(df_train), len(df_test))
            df_train.to_csv(self.training_file_name, index=False)
            df_test.to_csv(self.testing_file_name, index=False)
            
            preds = H2O_train_and_predict(self.training_file_name, self.testing_file_name, self.target_col, feat_cols)
            
            # append to result dataframe
            kwargs = {predict_col : preds[predict_col].values}
            df_test = df_test.assign (**kwargs)
            result = pd.concat([result, df_test[[predict_col, self.target_col]]])
            
            #remove_files ([training_file_name, testing_file_name])
            train_start += self.predict_interval
            
        return result


    def add_view (self, result_obj, view_name, dataframe, calc_error = False):
        result_obj[view_name] = {"data":dataframe}
        if calc_error:
            result_obj[view_name]["error"] = self.calc_AAE (dataframe)
        

    def calc_AAE (self, df):               # input:  two pd.Series
        df_tmp = df
        #df_tmp.index = range(len(df_tmp))        # quash the DatetimeIndex:  it causes problems
        actual = df_tmp[self.target_col]
        predictions = df_tmp["predict"]
        numerator = abs(predictions - actual)
        denominator = predictions.combine(actual, max)
        aae = numerator / denominator
        return aae.mean()

        
    def draw_charts_original (self, charts, filename):       
        ch_list = []            # TEMP !!!
        for chart in charts:
            df = charts[chart]["data"]
            title = self.target_col + ":  " + chart + "\n"
            title += filename
            if "error" in charts[chart]:
                title += "  (err=%s)" % charts[chart]["error"]
            title += "\nunit=" + self.resample_str + ", train=%dd, test=%dd" % (self.train_interval.days, self.predict_interval.days)
            outfile = self.compose_chart_name (filename, chart)
            draw_chart(title, df['predict'], df[self.target_col], df.index, outfile)
            ch_list.append ((title, df['predict'], df[self.target_col], df.index))     # TEMP !!!
        return ch_list

        
    def draw_charts (self, charts, filename):
        outfile = self.compose_chart_name2 (filename)
        ch_list = []            # TEMP !!!
        for chart in charts:
            df = charts[chart]["data"]
            title = self.target_col + ":  " + chart + "\n"
            #title += filename
            if "error" in charts[chart]:
                title += "Error=%s\n" % charts[chart]["error"]
            title += "unit=" + self.resample_str + ", train=%dd, test=%dd" % (self.train_interval.days, self.predict_interval.days)
            ch_list.append ((title, df['predict'], df[self.target_col], df.index))     # TEMP !!!
        draw_multi_charts (ch_list, filename, outfile)
        
        
    def compose_chart_name(self, entity, chart_type, subscriber=None):
        output_path = join(self.png_base, self.target_col)
        try:
            makedirs (output_path)
        except OSError:
            pass
        if subscriber:
            return join(output_path, subscriber) + "-" + entity + "-" + chart_type + ".png"
        else:
            return join(output_path, entity) + "-" + chart_type + ".png"
        

    def compose_chart_name2(self, entity):
        output_path = join(self.png_base, self.target_col)
        try:
            makedirs (output_path)
        except OSError:
            pass
        return join(output_path, entity) + ".png"
        
            
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
    



def get_page_dim(total):
    rows, cols = 1,1
    while rows * cols < total:
        if cols == rows:
            cols += 1
        else:
            rows += 1
    return rows, cols



def draw_multi_charts (chartlist, main_title, outputfile):
    fig = plt.figure(figsize=(11,8))
    rows, cols = get_page_dim (len(chartlist))
    index = 1
    
    for (chart_title, predicted, actual, dates) in chartlist:
        ax = fig.add_subplot(rows, cols, index)
        index += 1
        
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
        ax.set_title (chart_title)
        
    plt.title (main_title)
    fig.savefig(outputfile)
    plt.close()
    print (">>   wrote: ", outputfile)



    
        
if __name__ == "__main__":
    #fname = "VM_data/ad78e88c-ebb3-487e-ab6b-2eef62d81c5f.csv"
    #results = cp.process_file (fname)
    #cp.draw_charts(results, basename(fname))

    import argparse
    parser = argparse.ArgumentParser(description = "CROME training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target', help='target prediction column', default='cpu_usage')
    parser.add_argument('-d', '--dir', help = 'directory containing CSV files', type=str)
    parser.add_argument('-p', '--png_dir', help = 'destination directory for PNG files', default='./png')
    parser.add_argument('-n', '--max_files', help = 'process at most N files', type=int, default=1000000)
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    cfg = parser.parse_args()
    
    cp = CromeProcessor ('cpu_usage', png_base_path=cfg.png_dir)
    for fname in cfg.files[:cfg.max_files]:
        results = cp.process_file(fname)
        cp.draw_charts(results, basename(fname))
    
    
    
    

