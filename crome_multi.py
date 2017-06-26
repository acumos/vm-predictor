from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor        
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter

class SK_RFmodel:
    def __init__(self, estimators=20):
        self.estimators = estimators
        self.model = None
        self.features = None
        
    def train_and_predict(self, train_path, test_path, target_col, feat_cols, verbose=False):
        df_train = pd.read_csv(train_path)
        df_predict = pd.read_csv(test_path)
        rf = RandomForestRegressor(n_estimators=self.estimators)
        rf.fit(df_train[feat_cols], df_train[target_col])
        predicted = rf.predict(df_predict[feat_cols])
        return predicted
        
    def train (self, df_train, target_col, feat_cols):
        #df_train = pd.read_csv(train_path)
        rf = RandomForestRegressor(n_estimators=self.estimators)
        rf.fit(df_train[feat_cols], df_train[target_col])
        self.model = rf
        self.features = feat_cols
        return rf
        
    def predict (self, df_predict):
        #df_predict = pd.read_csv(predict_path)
        predicted = self.model.predict(df_predict[self.features])
        return predicted



        
def SK_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(df_train[feat_cols], df_train[target_col])
    predicted = rf.predict(df_predict[feat_cols])
    return predicted



def Scaler_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)
    x_scaler = StandardScaler().fit(df_train[feat_cols])
    Xt = x_scaler.transform(df_train[feat_cols])
    rf = RandomForestRegressor(n_estimators=20).fit(Xt, df_train[target_col])
    predicted = rf.predict(x_scaler.transform(df_predict[feat_cols]))
    return predicted
    

def ET_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)
    #rf = ExtraTreesRegressor(n_estimators=20)
    #rf = ExtraTreesRegressor(n_estimators=10)
    rf = ExtraTreesRegressor(n_estimators=5)
    rf.fit(df_train[feat_cols], df_train[target_col])
    predicted = rf.predict(df_predict[feat_cols])
    return predicted
    


def ETS_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)
    x_scaler = StandardScaler().fit(df_train[feat_cols])
    Xt = x_scaler.transform(df_train[feat_cols])
    rf = ExtraTreesRegressor(n_estimators=50).fit(Xt, df_train[target_col])
    predicted = rf.predict(x_scaler.transform(df_predict[feat_cols]))
    return predicted

    
    
def get_busy_hour (arr, period):
    best, high = None, None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x : x+period])
        if not high or score > high:
            best, high = x, score
    return arr.index[best]
          

def get_busy_avg (arr, period):
    best, high = None, None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x : x+period])
        if not high or score > high:
            best, high = x, score
    return high


  
          
          
class CromeProcessor(object):
    # See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for resample strings
    def __init__ (self, target_col, date_col='DATETIMEUTC', train_size_days=31, predict_size_days=1, resample_str="15min", 
                  png_base_path=".", min_train=300, feats=[], model=SK_RFmodel()):
        self.target_col = target_col
        self.predict_col = "predict"
        self.train_interval = pd.Timedelta(days=train_size_days)
        self.predict_interval = pd.Timedelta(days=predict_size_days)
        self.resample_str = resample_str
        self.png_base = png_base_path
        self.date_col=date_col
        self.min_train_rows = min_train
        self.min_predict_rows = 1
        self.pctile = 95
        self.training_file_name = "./train.csv"
        self.testing_file_name = "./test.csv"
        self.busyHours = 4        # in hours
        self.STD = True
        self.VAR = True
        self.showRaw = True
        self.smidgen = pd.Timedelta('1 second')
        self.one_hour = pd.Timedelta('1 hour')
        self.features = feats
        self.entity_col = 'VM_ID'
        self.max_entities = 10              # TEST VALUE !!!
        if len(self.features) < 1:  
            print ("CromeProcessor WARNING:  no features defined")
        self.model = model
        
        
            
    def ORIGINAL_process_CSVfile (self, filename):
        views = []
        df = pd.read_csv(filename)
        print (">> %s: %s rows" % (filename, len(df)))
        df = self.transform_dataframe (df)
        if self.check_valid_target(df):
            df_result = self.predict_sliding_windows (df)
            views = self.build_views (df_result)
        else:
            print (">>   Aborting:  all target values are identical.")
        return views


    def process_CSVfile (self, filename):
        big_df = pd.read_csv(filename)
        big_df = cleanup(big_df)
        print (">> %s: %s total rows" % (filename, len(big_df)))
        VM_list = sorted(list(set(big_df[self.entity_col])))
        if self.max_entities:
            VM_list = VM_list[:self.max_entities]
            big_df = big_df[big_df[self.entity_col].isin(VM_list)]
        df_result = self.predict_sliding_windows (big_df, VM_list)
        return df_result, VM_list

       
    def build_model_from_CSV (self, CSV_filename, datafile_out=None):
        df = pd.read_csv(CSV_filename)
        df = self.transform_dataframe (df)
        train_start = df.index[0]                              # TBD:  add optional start date
        train_stop = train_start + self.train_interval
        df = df[train_start : train_stop - self.smidgen]       # DatetimeIndex slices are inclusive
        if datafile_out:
            df.to_csv(datafile_out)
        return self.model.train (df, self.target_col, self.features)

       
    def predict_CSV (self, CSV_filename):
        df = pd.read_csv(CSV_filename)
        df = self.transform_dataframe (df)
        predict_start = df.index[0]
        predict_stop = predict_start + self.predict_interval
        df = df[predict_start : predict_stop - self.smidgen]    # DatetimeIndex slices are inclusive
        return self.model.predict(df)

        
    def add_derived_features (self, df):
        for feat in self.features:
            if feat == 'month':
                df[feat] = df.index.month
            elif feat == 'day':
                df[feat] = df.index.day
            elif feat == 'weekday':
                df[feat] = df.index.weekday
            elif feat == 'hour':
                df[feat] = df.index.hour
            elif feat == 'minute':
                df[feat] = df.index.minute
            elif feat.startswith ('hist-'):      # history features are of the form "hist-x" or "hist-x-y" where x and y are valid Timedelta strings such as '1H'
                params = feat.split("-")
                p1 = params[1]                   # first parameter (x) is the shift i.e. how long ago
                if len(params) > 2:
                    p2 = feat.split("-")[2]      # 2nd param (y) if present is the size of the window
                    df[feat] = df[self.target_col].shift(freq=pd.Timedelta(p1)).rolling(p2).mean()
                else:
                    df[feat] = df[self.target_col].shift(freq=pd.Timedelta (p1))
        return df
        
    
    def transform_dataframe (self, df):       # could be a pipeline
        # 1. convert to datetime index & sort
        DT = 'DT'
        df[DT] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(DT)
        df.index = pd.DatetimeIndex(df['DT'])    
        df.drop (self.date_col, axis=1, inplace=True)
        df.drop (DT, axis=1, inplace=True)
        
        # 2. re-sample at desired interval, e.g. "15min" or "1H".  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        if self.resample_str:
            df = df.resample (self.resample_str).mean().fillna(method='pad')
        
        # 3. get features
        df = self.add_derived_features(df)
        pre_drop = len(df)
        df.dropna(subset=self.features, inplace=True)
        if len(df) <  pre_drop/2:
            print (">>   Warning:  %s rows dropped during transform operation." % (pre_drop-len(df), )) 
        
        # TO DO :  drop all columns which are not target or features  ??
        return df


    def check_valid_target (self, df):
        vc = df[self.target_col].value_counts()
        return len(vc) >= 2


    def train_timeslice_model (self, master_df, VM_list, train_start, train_stop):
        train_data = pd.DataFrame()
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm].copy()
            print (">>      %s:  add %s rows " % (vm, len(df)))
            df = self.transform_dataframe (df)
            df = df[train_start : train_stop - self.smidgen]
            train_data = pd.concat ([train_data, df])
        bigmodel = None
        if len(train_data) >= self.min_train_rows and self.check_valid_target (train_data):
            bigmodel = self.model.train (train_data, self.target_col, self.features)
        return bigmodel
            

    def predict_timeslice_model (self, bigmodel, master_df, VM_list, predict_start, predict_stop):
        result = pd.DataFrame()
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm].copy()
            df = self.transform_dataframe (df)
            df = df[predict_start : predict_stop - self.smidgen]
            if len(df) >= self.min_predict_rows:
                preds = bigmodel.predict (df[self.features])
                # append to result dataframe
                df[self.predict_col] = preds
                df[self.entity_col] = vm
                result = pd.concat([result, df[[self.entity_col, self.predict_col, self.target_col]]])
        return result

    def ALTERNATE_predict_timeslice_model (self, bigmodel, master_df, VM_list, predict_start, predict_stop):
        result = {}
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm].copy()
            df = self.transform_dataframe (df)
            df = df[predict_start : predict_stop - self.smidgen]
            if len(df) >= self.min_predict_rows:
                preds = bigmodel.predict (df[self.features])
                df[self.predict_col] = preds
                result[vm] = df[[self.predict_col, self.target_col]]
        return result

            
    def predict_sliding_windows (self, master_df, VM_list):
        train_start = min(pd.to_datetime(master_df[self.date_col]))        # TBD:  allow user to specify start date
        result = pd.DataFrame()
        while True:
            # per day:  train ONE model for all VMs.   Then predict EACH VM separately.
            train_stop = train_start + self.train_interval
            predict_start = train_stop
            predict_stop = predict_start + self.predict_interval
            print (">>    train from %s to %s;  predict from %s to %s" % (train_start, train_stop, predict_start, predict_stop))
            xmodel = self.train_timeslice_model (master_df, VM_list, train_start, train_stop)
            if not xmodel:
                break
            preds = self.predict_timeslice_model (xmodel, master_df, VM_list, predict_start, predict_stop)
            result = pd.concat([result, preds])
            train_start += self.predict_interval
        return result


    def build_views (self, master_df, VM_list):
        output = []
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm]
            if len(df) > 0:
                if self.showRaw:
                    self.add_view (output, "original", df, vm, False)
                if self.pctile:
                    self.add_view (output, "percentile_%s" % self.pctile, df.resample("1D").apply(lambda x: np.nanpercentile(x, self.pctile)), vm, True)
                if self.STD:
                    self.add_view (output, "std", df.resample("1D").apply(lambda x: np.std(x)), vm, False)
                if self.VAR:
                    self.add_view (output, "variance", df.resample("1D").apply(lambda x: np.var(x)), vm, False)
                if self.busyHours:
                    df_hour = df.resample("1H").mean()
                    self.add_view (output, "busy_hour_%sH" % self.busyHours, df_hour.resample("1D").apply(lambda x: get_busy_hour(x, self.busyHours).hour), vm, False)
                    self.add_view (output, "busy_avg_%sH" % self.busyHours, df_hour.resample("1D").apply(lambda x: get_busy_avg(x, self.busyHours)), vm, True)
            else:
                print (">> build_views:  insufficient data for %s" % vm)
        return output
        

    def add_view (self, result_obj, view_name, dataframe, entity, calc_error = False):
        entry = {"entity": entity, "type":view_name, "data":dataframe}
        if calc_error:
            entry["error"] = self.calc_AAE (dataframe)        
        result_obj.append(entry)
        return result_obj
        

    def calc_AAE (self, df):               # input:  two pd.Series
        df_tmp = df
        #df_tmp.index = range(len(df_tmp))        # may need to quash the DatetimeIndex:  it sometimes causes problems
        actual = df_tmp[self.target_col]
        predictions = df_tmp[self.predict_col]
        numerator = abs(predictions - actual)
        denominator = predictions.combine(actual, max)
        aae = numerator / denominator
        return aae.mean()

            
    def draw_charts (self, charts):
        for chart in charts:
            df = chart["data"]
            title = self.target_col + ":  " + chart["type"]
            if "error" in chart:
                title += "  (err=%s)" % chart["error"]
            title += "\n" + chart["entity"] + " " + self.resample_str + " train %dd test %dd" % (self.train_interval.days, self.predict_interval.days)
            title += "\n[%s]" % " ".join(self.features)
            outfile = self.compose_chart_name (chart["entity"], chart["type"])
            if exists(outfile):
                print (">> %s:  chart already exists, skipped" % fname)
            draw_chart(title, df[self.predict_col], df[self.target_col], df.index, outfile)


    def draw_compound_chart (self, charts, filename):
        if len(charts) > 0:
            outfile = self.compose_chart_name (filename)
            ch_list = []
            for chart in charts:
                df = chart["data"]
                title = chart["type"]
                if "error" in chart:
                    title += " (err=%5.3f)" % chart["error"]
                ch_list.append ((title, df[self.predict_col], df[self.target_col], df.index))
            bigtitle = "%s %s %s train=%dd test=%dd" % (self.target_col, filename, self.resample_str, self.train_interval.days, self.predict_interval.days)
            bigtitle += "\n[%s]" % " ".join(self.features)
            draw_multi_charts (ch_list, bigtitle, outfile)


    def output_predictions (self, master_df, VM_list, skip_existing=False):
        master_df[self.date_col] = master_df.index
        for vm in VM_list:
            df_vm = master_df[master_df[self.entity_col]==vm]
            filepath = self.compose_file_name (vm, suffix="predict", extension=".csv")
            if exists (filepath):
                if skip_existing:
                    print ("  already exists")
                    continue
                # merge data w/ existing
                df_orig = pd.read_csv(filepath)
                df_vm = pd.concat([df_orig, df_vm])
            big_size = len(df_vm)
            df_vm = df_vm.drop_duplicates(subset=self.date_col)
            if len(df_vm) != big_size:
                print ("  dropped %s duplicate rows" % (big_size - len(df_vm),))
            df_vm[[self.date_col, self.predict_col, self.target_col]].to_csv(filepath, index=False)
            print (">>   wrote: ", filepath)

            
    def compose_chart_name(self, entity, chart_type="", subscriber=None):
        output_path = join(self.png_base, self.target_col)
        if chart_type:
            chart_type = "-" + chart_type
        try:
            makedirs (output_path)
        except OSError:
            pass
        if subscriber:
            return join(output_path, subscriber) + "-" + entity + chart_type + ".png"
        else:
            return join(output_path, entity) + chart_type + ".png"
        

    def compose_file_name(self, entity, suffix="", extension=".png"):
        output_path = join(self.png_base, self.target_col)
        if suffix:
            suffix = "-" + suffix
        try:
            makedirs (output_path)
        except OSError:
            pass
        return join(output_path, entity) + suffix + extension


        
            
def draw_chart (chart_title, predicted, actual, dates, png_filename):       # three pd.Series
    chart_width, chart_height = 11, 8.5
    fig = plt.figure(figsize=(chart_width,chart_height))
    day_count = 1 + (dates[-1] - dates[0]).days    
    ax = fig.add_subplot(111)
    
    ordinals = [matplotlib.dates.date2num(d) for d in dates] 
    ax.plot_date(ordinals,actual,'b-', label='Actual')
    ax.plot_date(ordinals,predicted,'r-', label='Predicted')

    ax.xaxis.set_major_locator(DayLocator(interval=compute_date_step(day_count, chart_width)))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
    locator = HourLocator()
    locator.MAXTICKS = (day_count + 3) * 24
    ax.xaxis.set_minor_locator(locator)

    ax.autoscale_view()
    ax.grid(True)
    fig.autofmt_xdate()
    legend = ax.legend(loc='upper right', shadow=True)
    plt.title (chart_title, fontsize=10)
    fig.savefig(png_filename)
    plt.close()
    print (">>   wrote: ", png_filename)

    
    
def draw_multi_charts (chartlist, main_title, outputfile):
    chart_width, chart_height = 14, 8.5
    fig = plt.figure(figsize=(chart_width,chart_height))
    rows, cols = get_page_dim (len(chartlist))
    index = 1

    for (chart_title, predicted, actual, dates) in chartlist:
        ax = fig.add_subplot(rows, cols, index)
        index += 1
        
        day_count = 1 + (dates[-1] - dates[0]).days    
        ordinals = [matplotlib.dates.date2num(d) for d in dates] 
        ax.plot_date(ordinals,actual,'b-', label='Actual')
        ax.plot_date(ordinals,predicted,'r-', label='Predicted')

        ax.xaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_major_locator(DayLocator(interval=compute_date_step(day_count, float(chart_width) / cols)))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
        locator = HourLocator()
        locator.MAXTICKS = (day_count + 3) * 24
        ax.xaxis.set_minor_locator(locator)
        
        ax.autoscale_view()
        ax.grid(True)
        fig.autofmt_xdate()
        
        legend = ax.legend(loc='upper right', shadow=True, fontsize=4)
        ax.set_title (chart_title)
        
    fig.suptitle(main_title)
    fig.savefig(outputfile)
    plt.close()
    print (">>   wrote: ", outputfile)


def get_page_dim(total):
    rows, cols = 1,1
    while rows * cols < total:
        if cols == rows:
            cols += 1
        else:
            rows += 1
    return rows, cols


def compute_date_step (day_count, chart_inches):
    optimal_dates_per_inch = 2.5
    d_p_i = day_count / chart_inches
    factor = d_p_i / optimal_dates_per_inch
    step = max (int(factor + 0.5), 1)
    return step
    



def remove_column_spaces (df):
    print (">> remove column spaces")
    replace_dict = {}
    for colname in df.columns:
        replace_dict[colname] = colname.replace(" ", "")
    df = df.rename(index=str, columns=replace_dict)
    return df



def trim_columns (df):
    col_list = list(df.select_dtypes(include=['object']).columns)
    for colname in col_list:
        print (">> trim: ", colname)
        try:
            df[colname] = df[colname].str.strip()
        except:
            pass
    return df


    
def cleanup (df):
    df = remove_column_spaces(df)
    df = trim_columns (df)              # remove leading and trailing spaces
    return df
    
                         



    
        
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description = "CROME training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target', help='target prediction column', default='cpu_usage')
    parser.add_argument('-c', '--compound', help = 'output compound charts', action='store_true')
    parser.add_argument('-s', '--separate', help = 'output separate charts', action='store_true')
    parser.add_argument('-r', '--randomize', help = 'randomize file list', action='store_true')
    parser.add_argument('-p', '--png_dir', help = 'destination directory for PNG files', default='./png')
    parser.add_argument('-n', '--max_files', help = 'process at most N files', type=int, default=1000000)
    parser.add_argument('-m', '--min_train', help = 'minimum # samples in a training set', type=int, default=300)
    parser.add_argument('-D', '--date_col', help='column to use for datetime index', default='DATETIMEUTC')
    parser.add_argument('-T', '--train_days', help = 'size of training set in days', type=int, default=31)
    parser.add_argument('-P', '--predict_days', help = 'number of days to predict per iteration', type=int, default=1)
    parser.add_argument('-S', '--sample_size', help='desired duration of train/predict units.  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases', default='15min')
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-f', '--features', nargs='+', help='list of features to use', default=['day', 'weekday', 'hour', 'minute'])
    parser.add_argument('-M', '--ML_platform', help='specify machine learning platform to use', default='SK')
    parser.add_argument('-w', '--write_predictions', help='write/merge predictions to PNG_DIR', action='store_true')
    
    
    cfg = parser.parse_args()

    if cfg.randomize:
        from random import shuffle
        shuffle (cfg.files)

    ML_model = None
    if cfg.ML_platform == "H2O":
        import ML_h2o
        ML_func = ML_h2o.H2O_train_and_predict
    elif cfg.ML_platform == "Scaler":
        ML_func = Scaler_train_and_predict
    elif cfg.ML_platform == "SK":
        ML_model = SK_RFmodel()
    elif cfg.ML_platform == "ET":
        ML_func = ET_train_and_predict
    elif cfg.ML_platform == "ETS":
        ML_func = ETS_train_and_predict
    elif cfg.ML_platform == "ARIMA":
        import ML_arima
        ML_func = ML_arima.ARIMA_train_and_predict

    cp = CromeProcessor (cfg.target, png_base_path=cfg.png_dir, date_col=cfg.date_col, train_size_days=cfg.train_days, predict_size_days=cfg.predict_days, 
                         resample_str=cfg.sample_size, min_train=cfg.min_train, feats=cfg.features, model=ML_model)
    
    for fname in cfg.files[:cfg.max_files]:
        results, VM_list = cp.process_CSVfile (fname)
        if cfg.write_predictions:
            cp.output_predictions (results, VM_list)
        if cfg.compound or cfg.separate:
            views = cp.build_views (results, VM_list)
            if cfg.compound:
                cp.draw_compound_chart(views, basename(fname))
            if cfg.separate:
                cp.draw_charts(views)
   

