from __future__ import print_function
print ("startup")

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor        
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter

from StringColumnEncoder import StringColumnEncoder

    
    
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
                  png_base_path=".", min_train=300, feats=[], max_entities=None, model=RandomForestRegressor()):
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
        self.max_entities = max_entities
        if len(self.features) < 1:  
            print ("CromeProcessor WARNING:  no features defined")
        self.model = model


    def process_CSVfiles (self, file_list):
        big_df, VM_list = self.preprocess_files(file_list)
        df_result = self.predict_sliding_windows (big_df, VM_list)
        return df_result, VM_list

        
    def build_model_from_CSV (self, file_list):                         # Note:  all files in the list will be appended
        master_df, VM_list = self.preprocess_files(file_list)
        train_start, range_end = self.find_time_range (master_df)       # TBD:  allow user to specify start/stop dates
        train_stop = train_start + self.train_interval
        xmodel = self.train_timeslice_model (master_df, VM_list, train_start, train_stop)
        return xmodel


    def predict_CSV (self, file_list, resample=None):
        self.resample_str = resample
        df, VM_list = self.preprocess_files(file_list)
        predict_start, predict_stop = self.find_time_range (df)       # TBD:  allow user to specify start/stop dates
        return self.predict_timeslice_model (self.model, df, VM_list, predict_start, predict_stop)
    
    
    def push_model(self, CSV_filelist, api):
        import cognita_client
        print (">> %s:  Loading raw features, training model" % CSV_filelist)
        model = self.build_model_from_CSV(CSV_filelist)
        print (">> %s:  Reload features, push to cognita" % CSV_filelist[0])        # if there's more than one file we push only the first
        df = pd.read_csv(CSV_filelist[0])
        try:
            cognita_client.push.push_sklearn_model(model, df[self.features],
                                                   extra_deps=None, api=api)
        except Exception as e:
            print(">> Error: Push error {:}".format(str(e.args[0])).encode("utf-8"))
            return False
        return True


    def preprocess_files (self, file_list):
        big_df = pd.DataFrame()
        for filename in file_list:
            print ("reading: ", filename)
            big_df = pd.concat([big_df, pd.read_csv(filename)])
        print (">> %s total rows" % len(big_df))
        big_df = cleanup(big_df)
        VM_list = sorted(list(set(big_df[self.entity_col])))
        if self.max_entities:
            print (">> applying max_entities= ", self.max_entities) 
            VM_list = VM_list[:self.max_entities]
            big_df = big_df[big_df[self.entity_col].isin(VM_list)]
        return big_df, VM_list
       
       
    def predict_sliding_windows (self, master_df, VM_list):
        train_start, range_end = self.find_time_range (master_df)       # TBD:  allow user to specify start/stop dates
        result = pd.DataFrame()
        while True:
            # per day:  train ONE model for all VMs.   Then predict EACH VM separately.
            train_stop = train_start + self.train_interval
            predict_start = train_stop
            if predict_start > range_end:
                break
            predict_stop = predict_start + self.predict_interval
            print (">>    train from %s to %s;  predict from %s to %s" % (train_start, train_stop, predict_start, predict_stop))
            xmodel = self.train_timeslice_model (master_df, VM_list, train_start, train_stop)
            if xmodel:
                preds = self.predict_timeslice_model (xmodel, master_df, VM_list, predict_start, predict_stop)
                result = pd.concat([result, preds])
                train_start += self.predict_interval
        return result
        

    def train_timeslice_model (self, master_df, VM_list, train_start, train_stop):
        train_data = pd.DataFrame()
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm].copy()
            #print (">>      %s:  add %s rows " % (vm, len(df)))
            df = self.transform_dataframe (df)
            df = df[train_start : train_stop - self.smidgen]       # DatetimeIndex slices are inclusive
            train_data = pd.concat ([train_data, df])
        bigmodel = None
        if len(train_data) >= self.min_train_rows and self.check_valid_target (train_data):
            print (">>      training %s entities %s total rows" % (len(VM_list), len(train_data)))
            bigmodel = self.model.fit (train_data[self.features], train_data[self.target_col])
        return bigmodel

        
    def predict_timeslice_model (self, bigmodel, master_df, VM_list, predict_start, predict_stop):
        result = pd.DataFrame()
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm].copy()
            df = self.transform_dataframe (df)
            df = df[predict_start : predict_stop - self.smidgen]       # DatetimeIndex slices are inclusive
            if len(df) >= self.min_predict_rows:
                preds = bigmodel.predict (df[self.features])
                # append to result dataframe
                df[self.predict_col] = preds
                df[self.entity_col] = vm
                result = pd.concat([result, df[[self.entity_col, self.predict_col, self.target_col]]])
        return result
        
            
    def transform_dataframe (self, df):       # could be a pipeline
        # 1. convert to datetime index & sort
        DT = 'DT'
        df[DT] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(DT)
        df.index = pd.DatetimeIndex(df[DT])
        df.drop (self.date_col, axis=1, inplace=True)
        df.drop (DT, axis=1, inplace=True)
        
        # 2. re-sample at desired interval, e.g. "15min" or "1H".  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        if self.resample_str:
            df = self.dual_resample (df)
        
        # 3. get features
        df = self.add_derived_features(df)
        pre_drop = len(df)
        df.dropna(subset=self.features, inplace=True)
        #if len(df) <  pre_drop/2:
        #    print (">>   Warning:  %s rows dropped during transform operation." % (pre_drop-len(df), )) 
        #4.  keep only feature and target columns
        df = df[self.features + [self.target_col]]
        return df

        
    def dual_resample (self, df):               # resample strings and numerics using separate methods.  This preserves the string columns.
        df_obj = df[df.columns[df.dtypes==object]]
        df_obj = df_obj.resample (self.resample_str).first().fillna(method='pad')
        df = df[df.columns[df.dtypes!=object]]
        df = df.resample (self.resample_str).mean().fillna(method='pad')
        for col in df_obj.columns:
            df[col] = df_obj[col]
        return df


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
        

    def check_valid_target (self, df):
        vc = df[self.target_col].value_counts()
        return len(vc) >= 2


    def find_time_range (self, df):
        print ("find time range...")
        times = pd.to_datetime(df[self.date_col])
        range_start, range_end = min(times), max(times)        
        print ("  %s to %s" % (range_start, range_end))
        return range_start, range_end
        
        
    def build_views (self, master_df, VM_list):
        output = []
        for vm in VM_list:
            df = master_df[master_df[self.entity_col]==vm]
            self.build_VM_views (df, vm, output)
        return output
        

    def build_VM_views (self, df, vm, output):
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

            
    def draw_charts (self, chart_list):
        for chart in chart_list:
            df = chart["data"]
            title = self.target_col + ":  " + chart["type"]
            if "error" in chart:
                title += "  (err=%s)" % chart["error"]
            title += "\n" + chart["entity"] + " " + self.resample_str + " train %dd test %dd" % (self.train_interval.days, self.predict_interval.days)
            title += "\n[%s]" % " ".join(self.features)
            outfile = self.compose_chart_name (chart["entity"], chart["type"])
            if outfile:
                draw_chart(title, df[self.predict_col], df[self.target_col], df.index, outfile)


    def draw_compound_charts (self, chart_list):
        if len(chart_list) > 0:
            # Incoming charts may contain multiple entities.  Group by entity first.
            from collections import defaultdict
            grouped=defaultdict(list)  
            for chart in chart_list:  
                grouped[chart['entity']].append(chart)

            # Now draw compound charts for each entity
            for entity, charts in grouped.items():              # python 3 syntax
                outfile = self.compose_chart_name (entity)
                if not outfile:
                    continue
                ch_list = []
                for chart in charts:
                    df = chart["data"]
                    title = chart["type"]
                    if "error" in chart:
                        title += " (err=%5.3f)" % chart["error"]
                    ch_list.append ((title, df[self.predict_col], df[self.target_col], df.index))
                bigtitle = "%s %s %s train=%dd test=%dd" % (self.target_col, entity, self.resample_str, self.train_interval.days, self.predict_interval.days)
                bigtitle += "\n[%s]" % " ".join(self.features)
                draw_multi_charts (ch_list, bigtitle, outfile)


    def output_predictions (self, master_df, VM_list):
        import json
        for vm in VM_list:
            df_vm = master_df[master_df[self.entity_col]==vm]
            filepath = self.compose_file_name (vm, suffix="predict", extension=".json", skip_existing=False)
            if exists (filepath):
                # merge data w/ existing
                with open(filepath) as fp:
                    j_dict = json.load(fp)
                df_orig = pd.read_json(j_dict['data'], orient='index')                
                df_vm = pd.concat([df_orig, df_vm])
            big_size = len(df_vm)
            df_vm = df_vm[~df_vm.index.duplicated(keep='first')]
            if len(df_vm) != big_size:
                print ("  dropped %s duplicate rows" % (big_size - len(df_vm),))
            jdata = df_vm.to_json(orient='index')
            record = {'entity':vm, 'features':self.features, 'target':self.target_col, 'interval':self.resample_str, 'train_days':self.train_interval.days, 'predict_days':self.predict_interval.days, 'data':jdata}
            with open(filepath, "w") as fp:
                json.dump(record, fp)
            print (">>   wrote: ", filepath)

       
    def compose_chart_name(self, entity, chart_type="", subscriber=None):
        return self.compose_file_name (entity, suffix=chart_type)
    

    def compose_file_name(self, entity, suffix="", extension=".png", skip_existing=True):
        output_path = join(self.png_base, self.target_col)
        if suffix:
            suffix = "-" + suffix
        try:
            makedirs (output_path)
        except OSError:
            pass
        filename = join(output_path, entity) + suffix + extension
        if skip_existing and exists(filename):
            print ("  skip existing file: ", filename)
            filename = None
        return filename


        
            
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

    
def cleanup (df):
    df = remove_column_spaces(df)
    df = trim_columns (df)              # remove leading and trailing spaces
    return df
   
    

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


    
    
                         

    
        
if __name__ == "__main__":

    print ("main")
    import argparse
    parser = argparse.ArgumentParser(description = "CROME training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target', help='target prediction column', default='cpu_usage')
    parser.add_argument('-c', '--compound', help = 'output compound charts', action='store_true')
    parser.add_argument('-s', '--separate', help = 'output separate charts', action='store_true')
    parser.add_argument('-r', '--randomize', help = 'randomize file list', action='store_true')
    parser.add_argument('-o', '--output_dir', help = 'destination directory for output files', default='./results')
    parser.add_argument('-p', '--write_predictions', help='write/merge predictions to OUTPUT_DIR', action='store_true')
    parser.add_argument('-n', '--max_files', help = 'open at most N files', type=int, default=1000000)
    parser.add_argument('-m', '--min_train', help = 'minimum # samples in a training set', type=int, default=300)
    parser.add_argument('-D', '--date_col', help='column to use for datetime index', default='DATETIMEUTC')
    parser.add_argument('-T', '--train_days', help = 'size of training set in days', type=int, default=31)
    parser.add_argument('-P', '--predict_days', help = 'number of days to predict per iteration', type=int, default=1)
    parser.add_argument('-S', '--sample_size', help='desired duration of train/predict units.  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases', default='15min')
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-f', '--features', nargs='+', help='list of features to use', default=['month', 'day', 'weekday', 'hour', 'minute'])
    parser.add_argument('-M', '--ML_type', help='specify machine learning model type to use', default='RF')
    parser.add_argument('-j', '--join_files', help='process list of files as one', action='store_true')
    parser.add_argument('-v', '--max_entities', help = 'process at most N entities (VMs)', type=int, default=None)
    parser.add_argument('-i', '--set_param', help='set ML model integer parameter', action='append', nargs=2, default=[])
    parser.add_argument('-a', '--push_address', help='server address to push the model', default='')
    
    
    cfg = parser.parse_args()

    if cfg.randomize:
        from random import shuffle
        shuffle (cfg.files)

    ML_model = None
    if cfg.ML_type == "H2O":
        import ML_h2o
        ML_func = ML_h2o.H2O_train_and_predict
    elif cfg.ML_type == "RF_SC":
        ML_model = Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20))])
    elif cfg.ML_type == "RF":
        ML_model = Pipeline([('enc', StringColumnEncoder()), ('rf', RandomForestRegressor(n_estimators=20))])
    elif cfg.ML_type == "ET":
        ML_model = Pipeline([('enc', StringColumnEncoder()), ('et', ExtraTreesRegressor(n_estimators=20))]) 
    elif cfg.ML_type == "ET_SC":
        ML_model = Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('et', ExtraTreesRegressor(n_estimators=20))])
    elif cfg.ML_type == "ARIMA":
        import ML_arima
        ML_func = ML_arima.ARIMA_train_and_predict

    # support for model-specific parameters
    if len(cfg.set_param) > 0:
        param_dict = {k:int(v) for [k,v] in cfg.set_param}
        ML_model.set_params (**param_dict)
        print ("set model params:", param_dict)
        
    print ("constructor")
    cp = CromeProcessor (cfg.target, png_base_path=cfg.output_dir, date_col=cfg.date_col, train_size_days=cfg.train_days, predict_size_days=cfg.predict_days, 
                         resample_str=cfg.sample_size, min_train=cfg.min_train, feats=cfg.features, max_entities=cfg.max_entities, model=ML_model)
                         
    if cfg.join_files:
        file_list = [cfg.files[:cfg.max_files]]
    else:
        file_list = [[x] for x in cfg.files[:cfg.max_files]]
        
    for fnames in file_list:
        if len(cfg.push_address)!=0:
            cp.push_model(fnames, cfg.push_address)
        else:
            results, VM_list = cp.process_CSVfiles (fnames)
            if cfg.write_predictions:
                cp.output_predictions (results, VM_list)
            if cfg.compound or cfg.separate:
                views = cp.build_views (results, VM_list)
                if cfg.compound:
                    cp.draw_compound_charts(views)
                if cfg.separate:
                    cp.draw_charts(views)
   
