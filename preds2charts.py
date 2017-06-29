import pandas as pd

from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter



from crome_multi import CromeProcessor


def convert_to_datetime_index (df, date_col):
    DT = 'DT'
    df[DT] = pd.to_datetime(df[date_col])
    df = df.sort_values(DT)
    df.index = pd.DatetimeIndex(df['DT'])    
    df.drop (date_col, axis=1, inplace=True)
    df.drop (DT, axis=1, inplace=True)
    return df


if __name__ == "__main__":

    print ("main")
    import argparse
    parser = argparse.ArgumentParser(description = "Draw charts from CROME prediction files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target', help='target prediction column', default='cpu_usage')
    parser.add_argument('-c', '--compound', help = 'generate compound charts', action='store_true')
    parser.add_argument('-s', '--separate', help = 'generate separate charts', action='store_true')
    parser.add_argument('-d', '--output_dir', help = 'destination directory for output files', default='./results')
    parser.add_argument('-n', '--max_files', help = 'open at most N files', type=int, default=1000000)
    parser.add_argument('-D', '--date_col', help='column to use for datetime index', default='DATETIMEUTC')
    parser.add_argument('-T', '--train_days', help = 'size of training set in days', type=int, default=31)
    parser.add_argument('-P', '--predict_days', help = 'number of days predicted per iteration', type=int, default=1)
    parser.add_argument('-S', '--sample_size', help='desired duration of train/predict units.  See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases', default='15min')
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-f', '--features', nargs='+', help='list of features used', default=['day', 'weekday', 'hour', 'minute'])
    parser.add_argument('-M', '--ML_platform', help='specify machine learning platform used', default='SK')
    parser.add_argument('-v', '--max_entities', help = 'process at most N entities (VMs)', type=int, default=10)
    
    cfg = parser.parse_args()
    cp = CromeProcessor (cfg.target, png_base_path=cfg.output_dir, date_col=cfg.date_col, train_size_days=cfg.train_days, predict_size_days=cfg.predict_days, 
                         resample_str=cfg.sample_size, feats=cfg.features, max_entities=cfg.max_entities)

    for fname in cfg.files:
        df = pd.read_csv(fname)
        df = convert_to_datetime_index (df, cp.date_col)
        views = []
        cp.build_VM_views (df, basename(fname), views)
        if cfg.separate:
            cp.draw_charts(views)
        if cfg.compound:
            cp.draw_compound_chart(views, basename(fname))
            
            
