from __future__ import print_function

import warnings
import itertools
import pandas as pd
#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


def make_dtindex (df):
    DT = 'DT'
    date_col = 'DATETIMEUTC'
    df[DT] = pd.to_datetime(df[date_col])
    df = df.sort_values(DT)
    df.index = pd.DatetimeIndex(df['DT'])    
    return df


def OLD_ARIMA_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)

    df_train.index = pd.DatetimeIndex(df_train['DT'])
    df_predict.index = pd.DatetimeIndex(df_predict['DT'])
    
    #rf = RandomForestRegressor(n_estimators=20)
    model = ARIMA(df_train[target_col], order=(5, 1, 5))
    #rf.fit(df_train[feat_cols], df_train[target_col])
    #import pdb; pdb.set_trace()
    model = model.fit(disp=0)
    #predicted = rf.predict(df_predict[feat_cols])
    predicted = model.predict(df_predict[feat_cols])
    return predicted

    
def ARIMA_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    #  from https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
    df_train = pd.read_csv(train_path)
    df_predict = pd.read_csv(test_path)

    df_train.index = pd.DatetimeIndex(df_train['DT'])
    df_predict.index = pd.DatetimeIndex(df_predict['DT'])
    
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    lowest_aic = 1000000000
    best_params = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df_train[target_col],
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False, verbose=0)

                results = mod.fit()
                if results.aic < lowest_aic:
                    lowest_aic = results.aic
                    best_params = (param, param_seasonal)

                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                #print(param, param_seasonal, results.aic)
            except:
                continue    
                
    print ("Best params: ", best_params)

    # Now (re-) load the model w/ best performance and make predictions.
    mod = sm.tsa.statespace.SARIMAX(df_train[target_col],
                                    order=best_params[0],
                                    seasonal_order=best_params[1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False, verbose=0)
    
    results = mod.fit()
    p_start = df_train.index[-1]            # strangely enough predict() only works if one of the dates is in the training range
    p_end = df_predict.index[-1]
    pred = results.predict(start=p_start, end=p_end)
    
    return pred[1:]
    