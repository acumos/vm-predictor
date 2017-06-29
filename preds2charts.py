import pandas as pd

from os import listdir, makedirs
from os.path import isfile, join, basename, exists
import matplotlib 
matplotlib.use ("Agg")
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter

import argparse
import json

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
    parser = argparse.ArgumentParser(description = "Draw charts from CROME prediction files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--compound', help = 'generate compound charts', action='store_true')
    parser.add_argument('-s', '--separate', help = 'generate separate charts', action='store_true')
    parser.add_argument('-o', '--output_dir', help = 'destination directory for output files', default='./results')
    parser.add_argument('-n', '--max_files', help = 'open at most N files', type=int, default=1000000)
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-v', '--max_entities', help = 'process at most N entities (VMs)', type=int, default=10)
    
    cfg = parser.parse_args()

    for fname in cfg.files:
        with open(fname) as fp:
            jdict = json.load(fp)
        df = pd.read_json(jdict['data'], orient='index')                
        cp = CromeProcessor (jdict['target'], png_base_path=cfg.output_dir, train_size_days=jdict['train_days'], predict_size_days=jdict['predict_days'], 
                             resample_str=jdict['interval'], feats=jdict['features'], max_entities=cfg.max_entities)
        views = []
        cp.build_VM_views (df, jdict['entity'], views)
        if cfg.separate:
            cp.draw_charts(views)
        if cfg.compound:
            cp.draw_compound_charts(views)
            
            
