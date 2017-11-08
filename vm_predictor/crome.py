from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from os import makedirs
from os.path import join, basename, exists

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter


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

    def train(self, df_train, target_col, feat_cols):
        # df_train = pd.read_csv(train_path)
        rf = RandomForestRegressor(n_estimators=self.estimators)
        rf.fit(df_train[feat_cols], df_train[target_col])
        self.model = rf
        self.features = feat_cols
        return rf

    def predict(self, df_predict):
        # df_predict = pd.read_csv(predict_path)
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
    # rf = ExtraTreesRegressor(n_estimators=20)
    # rf = ExtraTreesRegressor(n_estimators=10)
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


def get_busy_hour(arr, period):
    best, high = None, None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x:x + period])
        if not high or score > high:
            best, high = x, score
    return arr.index[best]


def get_busy_avg(arr, period):
    best, high = None, None
    for x in range(len(arr)):
        score = np.mean(arr.iloc[x:x + period])
        if not high or score > high:
            best, high = x, score
    return high


class CromeProcessor(object):
    # See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for resample strings
    def __init__(self, target_col, date_col='DATETIMEUTC', train_size_days=31, predict_size_days=1, resample_str="15min",
                 png_base_path=".", min_train=300, feats=[], model=SK_RFmodel()):
        self.target_col = target_col
        self.predict_col = "predict"
        self.train_interval = pd.Timedelta(days=train_size_days)
        self.predict_interval = pd.Timedelta(days=predict_size_days)
        self.resample_str = resample_str
        self.png_base = png_base_path
        self.date_col = date_col
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
        if len(self.features) < 1:
            print("CromeProcessor WARNING:  no features defined")
        self.model = model

    def process_CSVfile(self, filename):
        views = []
        df = pd.read_csv(filename)
        print(">> %s: %s rows" % (filename, len(df)))
        df = self.transform_dataframe(df)
        if self.check_valid_target(df):
            df_result = self.predict_sliding_windows(df)
            views = self.build_views(df_result)
        else:
            print(">>   Aborting:  all target values are identical.")
        return views

    def build_model_from_CSV(self, CSV_filename, datafile_out=None, is_raw_data=True):
        df = pd.read_csv(CSV_filename)
        if is_raw_data:
            df = self.transform_dataframe(df)
            train_start = df.index[0]                              # TBD:  add optional start date
            train_stop = train_start + self.train_interval
            df = df[train_start:train_stop - self.smidgen]       # DatetimeIndex slices are inclusive
        if datafile_out:
            df.to_csv(datafile_out)
        return self.model.train(df, self.target_col, self.features)

    def predict_CSV(self, CSV_filename):
        df = pd.read_csv(CSV_filename)
        df = self.transform_dataframe(df)
        predict_start = df.index[0]
        predict_stop = predict_start + self.predict_interval
        df = df[predict_start:predict_stop - self.smidgen]    # DatetimeIndex slices are inclusive
        return self.model.predict(df)

    def generate_model(self, CSV_filename, is_raw_data=False):
        from acumos.modeling import Model, List, create_namedtuple
        from acumos.session import Requirements
        from os import path

        print(">> %s:  Loading raw features, training model" % CSV_filename)
        model = self.build_model_from_CSV(CSV_filename, is_raw_data=is_raw_data)
        print(">> %s:  Reload features, push to server" % CSV_filename)
        df = pd.read_csv(CSV_filename)[self.features]
        listVars = [(df.columns[i], df.dtypes[i].type) for i in range(len(df.columns))]
        VmPredictorDataFrame = create_namedtuple('VmPredictorDataFrame', listVars)

        def predict_metric(df: VmPredictorDataFrame) -> List[float]:
            '''Returns an array of float predictions'''
            X = np.column_stack(df)
            return model.predict(X)

        # compute path of this package to add it as a dependency
        package_path = path.dirname(path.realpath(__file__))
        return Model(classify=predict_metric), Requirements(packages=[package_path])

    def push_model(self, CSV_filename, push_api, auth_api, is_raw_data=False):
        from acumos.session import AcumosSession
        session = AcumosSession(push_api=push_api, auth_api=auth_api)
        model, reqs = self.generate_model(CSV_filename, is_raw_data)
        try:
            session.push(model, 'VmPredictor', reqs)  # creates ./my-iris.zip
            return True
        except Exception as e:
            print(">> Error: Model push error {:}".format(e))
        return False

    def dump_model(self, CSV_filename, model_dir, is_raw_data=False):
        from acumos.session import AcumosSession
        model, reqs = self.generate_model(CSV_filename, is_raw_data)

        session = AcumosSession()
        try:
            session.dump(model, 'VmPredictor', model_dir, reqs)  # creates ./my-iris.zip
            return True
        except Exception as e:
            print(">> Error: Model dump error {:}".format(e))
        return False

    def add_derived_features(self, df):
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
            elif feat.startswith('hist-'):      # history features are of the form "hist-x" or "hist-x-y" where x and y are valid Timedelta strings such as '1H'
                params = feat.split("-")
                p1 = params[1]                   # first parameter (x) is the shift i.e. how long ago
                if len(params) > 2:
                    p2 = feat.split("-")[2]      # 2nd param (y) if present is the size of the window
                    df[feat] = df[self.target_col].shift(freq=pd.Timedelta(p1)).rolling(p2).mean()
                else:
                    df[feat] = df[self.target_col].shift(freq=pd.Timedelta(p1))
        return df

    def transform_dataframe(self, df):       # could be a pipeline
        # 1. convert to datetime index & sort
        DT = 'DT'
        df[DT] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(DT)
        df.index = pd.DatetimeIndex(df['DT'])
        df.drop(self.date_col, axis=1, inplace=True)
        df.drop(DT, axis=1, inplace=True)

        # 2. re-sample at desired interval, e.g. "15min" or "1H".  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        if self.resample_str:
            df = df.resample(self.resample_str).mean().fillna(method='pad')

        # 3. get features
        df = self.add_derived_features(df)
        pre_drop = len(df)
        df.dropna(subset=self.features, inplace=True)
        if len(df) < pre_drop / 2:
            print(">>   Warning:  %s rows dropped during transform operation." % (pre_drop - len(df), ))

        # TO DO :  drop all columns which are not target or features  ??
        return df

    def check_valid_target(self, df):
        vc = df[self.target_col].value_counts()
        if len(vc) < 2:
            return False
        # this test is irrelevant for regressors
        # valid = 0
        # for idx,val in vc.iteritems():
        #    if val > 1:
        #        valid += 1
        #        if valid > 1:
        #            return True
        # return False
        return True

    def predict_sliding_windows(self, df):        # presumed that the incoming dataframe contains ONLY rows of interest
        train_start = df.index[0]
        result = pd.DataFrame()
        while True:
            train_stop = train_start + self.train_interval
            predict_start = train_stop
            predict_stop = predict_start + self.predict_interval

            df_train = df[train_start:train_stop - self.smidgen]       # DatetimeIndex slices are inclusive
            if len(df_train) < self.min_train_rows:
                break

            df_test = df[predict_start:predict_stop - self.smidgen]    # DatetimeIndex slices are inclusive
            if len(df_test) < self.min_predict_rows:
                break

            print(">>   train %s rows from %s to %s;  predict %s rows from %s to %s" % (len(df_train), train_start, train_stop, len(df_test), predict_start, predict_stop))
            if self.check_valid_target(df_train):  # and self.check_valid_target (df_test):
                df_train.to_csv(self.training_file_name, index=True)
                df_test.to_csv(self.testing_file_name, index=True)

                preds = self.model.train_and_predict(self.training_file_name, self.testing_file_name, self.target_col, self.features)

                # append to result dataframe
                kwargs = {self.predict_col: preds}
                df_test = df_test.assign(**kwargs)
                result = pd.concat([result, df_test[[self.predict_col, self.target_col]]])
            else:
                print(">>   invalid data")

            # remove_files ([training_file_name, testing_file_name])   ??
            train_start += self.predict_interval

        return result

    def build_views(self, df):
        output = []
        if len(df) > 0:
            if self.showRaw:
                self.add_view(output, "original", df, False)
            if self.pctile:
                self.add_view(output, "percentile_%s" % self.pctile, df.resample("1D").apply(lambda x: np.nanpercentile(x, self.pctile)), True)
            if self.STD:
                self.add_view(output, "std", df.resample("1D").apply(lambda x: np.std(x)), False)
            if self.VAR:
                self.add_view(output, "variance", df.resample("1D").apply(lambda x: np.var(x)), False)
            if self.busyHours:
                df_hour = df.resample("1H").mean()
                self.add_view(output, "busy_hour_%sH" % self.busyHours, df_hour.resample("1D").apply(lambda x: get_busy_hour(x, self.busyHours).hour), False)
                self.add_view(output, "busy_avg_%sH" % self.busyHours, df_hour.resample("1D").apply(lambda x: get_busy_avg(x, self.busyHours)), True)
        else:
            print(">> build_views:  insufficient data")
        return output

    def add_view(self, result_obj, view_name, dataframe, calc_error=False):
        entry = {"type": view_name, "data": dataframe}
        if calc_error:
            entry["error"] = self.calc_AAE(dataframe)
        result_obj.append(entry)
        return result_obj

    def calc_AAE(self, df):               # input:  two pd.Series
        df_tmp = df
        # df_tmp.index = range(len(df_tmp))        # may need to quash the DatetimeIndex:  it sometimes causes problems
        actual = df_tmp[self.target_col]
        predictions = df_tmp[self.predict_col]
        numerator = abs(predictions - actual)
        denominator = predictions.combine(actual, max)
        aae = numerator / denominator
        return aae.mean()

    def draw_charts(self, charts, filename):
        if filename is None:
            return
        for chart in charts:
            df = chart["data"]
            title = self.target_col + ":  " + chart["type"]
            if "error" in chart:
                title += "  (err=%s)" % chart["error"]
            title += "\n" + filename + " " + self.resample_str + " train %dd test %dd" % (self.train_interval.days, self.predict_interval.days)
            title += "\n[%s]" % " ".join(self.features)
            outfile = self.compose_chart_name(filename, chart["type"])
            draw_chart(title, df[self.predict_col], df[self.target_col], df.index, outfile)

    def draw_compound_chart(self, charts, filename):
        if filename is None:
            return
        if len(charts) > 0:
            outfile = self.compose_chart_name(filename)
            ch_list = []
            for chart in charts:
                df = chart["data"]
                title = chart["type"]
                if "error" in chart:
                    title += " (err=%5.3f)" % chart["error"]
                ch_list.append((title, df[self.predict_col], df[self.target_col], df.index))
            bigtitle = "%s %s %s train=%dd test=%dd" % (self.target_col, filename, self.resample_str, self.train_interval.days, self.predict_interval.days)
            bigtitle += "\n[%s]" % " ".join(self.features)
            draw_multi_charts(ch_list, bigtitle, outfile)

    def compose_chart_name(self, entity, chart_type="", subscriber=None):
        if len(self.png_base) == 0:
            return None
        output_path = join(self.png_base, self.target_col)
        if chart_type:
            chart_type = "-" + chart_type
        try:
            makedirs(output_path)
        except OSError:
            pass
        if subscriber:
            return join(output_path, subscriber) + "-" + entity + chart_type + ".png"
        else:
            return join(output_path, entity) + chart_type + ".png"


def draw_chart(chart_title, predicted, actual, dates, png_filename):  # three pd.Series
    if png_filename is None:
        return
    chart_width, chart_height = 11, 8.5
    fig = plt.figure(figsize=(chart_width, chart_height))
    day_count = 1 + (dates[-1] - dates[0]).days
    ax = fig.add_subplot(111)

    ordinals = [matplotlib.dates.date2num(d) for d in dates]
    ax.plot_date(ordinals, actual, 'b-', label='Actual')
    ax.plot_date(ordinals, predicted, 'r-', label='Predicted')

    ax.xaxis.set_major_locator(DayLocator(interval=compute_date_step(day_count, chart_width)))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
    locator = HourLocator()
    locator.MAXTICKS = (day_count + 3) * 24
    ax.xaxis.set_minor_locator(locator)

    ax.autoscale_view()
    ax.grid(True)
    fig.autofmt_xdate()
    ax.legend(loc='upper right', shadow=True)
    plt.title(chart_title, fontsize=10)
    fig.savefig(png_filename)
    plt.close()
    print(">>   wrote: ", png_filename)


def draw_multi_charts(chartlist, main_title, outputfile):
    if outputfile is None:
        return
    chart_width, chart_height = 14, 8.5
    fig = plt.figure(figsize=(chart_width, chart_height))
    rows, cols = get_page_dim(len(chartlist))
    index = 1

    for (chart_title, predicted, actual, dates) in chartlist:
        ax = fig.add_subplot(rows, cols, index)
        index += 1

        day_count = 1 + (dates[-1] - dates[0]).days
        ordinals = [matplotlib.dates.date2num(d) for d in dates]
        ax.plot_date(ordinals, actual, 'b-', label='Actual')
        ax.plot_date(ordinals, predicted, 'r-', label='Predicted')

        ax.xaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_major_locator(DayLocator(interval=compute_date_step(day_count, float(chart_width) / cols)))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
        locator = HourLocator()
        locator.MAXTICKS = (day_count + 3) * 24
        ax.xaxis.set_minor_locator(locator)

        ax.autoscale_view()
        ax.grid(True)
        fig.autofmt_xdate()

        ax.legend(loc='upper right', shadow=True, fontsize=4)
        ax.set_title(chart_title)

    fig.suptitle(main_title)
    fig.savefig(outputfile)
    plt.close()
    print(">>   wrote: ", outputfile)


def get_page_dim(total):
    rows, cols = 1, 1
    while rows * cols < total:
        if cols == rows:
            cols += 1
        else:
            rows += 1
    return rows, cols


def compute_date_step(day_count, chart_inches):
    optimal_dates_per_inch = 2.5
    d_p_i = day_count / chart_inches
    factor = d_p_i / optimal_dates_per_inch
    step = max(int(factor + 0.5), 1)
    return step


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VM Predictor training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target', help='target prediction column', default='cpu_usage')
    parser.add_argument('-c', '--compound', help='output compound charts', action='store_true')
    parser.add_argument('-s', '--separate', help='output separate charts', action='store_true')
    parser.add_argument('-r', '--randomize', help='randomize file list', action='store_true')
    parser.add_argument('-p', '--png_dir', help='destination directory for PNG files', default='')
    parser.add_argument('-n', '--max_files', help='process at most N files', type=int, default=1000000)
    parser.add_argument('-m', '--min_train', help='minimum # samples in a training set', type=int, default=300)
    parser.add_argument('-D', '--date_col', help='column to use for datetime index', default='DATETIMEUTC')
    parser.add_argument('-T', '--train_days', help='size of training set in days', type=int, default=31)
    parser.add_argument('-P', '--predict_days', help='number of days to predict per iteration', type=int, default=1)
    parser.add_argument('-S', '--sample_size', help='desired duration of train/predict units.  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases', default='15min')
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-f', '--features', nargs='+', help='list of features to use', default=['day', 'weekday', 'hour', 'minute'])
    parser.add_argument('-M', '--ML_platform', help='specify machine learning platform to use', default='RF')
    parser.add_argument('-R', '--is_raw_data', help='for the push and dump options, perform feature processing', default=False, action='store_true')
    parser.add_argument('-a', '--push_address', help='server address to push the model', default='')
    parser.add_argument('-A', '--auth_address', help='server address for login and push of the model', default='')
    parser.add_argument('-d', '--dump_pickle', help='dump model to a pickle directory for local running', default='')

    cfg = parser.parse_args()

    if cfg.randomize:
        from random import shuffle
        shuffle(cfg.files)

    ML_model = None
    if cfg.ML_platform == "RF":
        ML_model = SK_RFmodel()
    """
    disabling alternate work right now...
    elif cfg.ML_platform == "H2O":
        from vm_predictor import ML_h2o
        ML_func = ML_h2o.H2O_train_and_predict
    elif cfg.ML_platform == "Scaler":
        ML_func = Scaler_train_and_predict
    elif cfg.ML_platform == "ET":
        ML_func = ET_train_and_predict
    elif cfg.ML_platform == "ETS":
        ML_func = ETS_train_and_predict
    elif cfg.ML_platform == "ARIMA":
        from vm_predictor import ML_arima
        ML_func = ML_arima.ARIMA_train_and_predict
    """

    cp = CromeProcessor(cfg.target, png_base_path=cfg.png_dir, date_col=cfg.date_col, train_size_days=cfg.train_days,
                        predict_size_days=cfg.predict_days, resample_str=cfg.sample_size, min_train=cfg.min_train,
                        feats=cfg.features, model=ML_model)

    for fname in cfg.files[:cfg.max_files]:
        if len(cfg.push_address) != 0:
            cp.push_model(fname, cfg.push_address, cfg.auth_address, cfg.is_raw_data)
        elif len(cfg.dump_pickle) != 0:
            cp.dump_model(fname, cfg.dump_pickle, cfg.is_raw_data)
        elif len(cfg.png_dir) != 0 and (exists(cp.compose_chart_name(basename(fname))) or exists(cp.compose_chart_name(basename(fname), 'original'))):
            print(">> %s:  chart already exists, skipped" % fname)
        else:
            results = cp.process_CSVfile(fname)
            if cfg.compound:
                if len(cfg.push_address) != 0:
                    cp.push_model(fname, cfg.push_address, cfg.auth_address, cfg.is_raw_data)
                elif len(cfg.dump_pickle) != 0:
                    cp.dump_model(fname, cfg.dump_pickle, cfg.is_raw_data)
                cp.draw_compound_chart(results, basename(fname))
            if cfg.separate:
                cp.draw_charts(results, basename(fname))


if __name__ == "__main__":
    main()
