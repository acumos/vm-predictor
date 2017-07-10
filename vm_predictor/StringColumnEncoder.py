import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class SimpleEncoder:
    def __init__(self, unknown_label=-1):
        self.lookup = {}
        self.classes_ = []
        self.unknown = unknown_label
        
    def fit(self, X):
        self.classes_ = []
        unique = sorted(set(X))
        for idx,val in enumerate(unique):
            self.lookup[val] = idx
            self.classes_.append (val)

    def transform(self, X):
        out = []
        for k in X:
            try:
                val = self.lookup[k]
            except: 
                val = self.unknown
            out.append(val)
        return np.array(out)
        

        
class StringColumnEncoder (BaseEstimator, TransformerMixin):
    def __init__ (self):
        self.encoder = defaultdict(SimpleEncoder)
        
    def fit (self, X, y=None):
        df_obj = X[X.columns[X.dtypes==object]]                     # apply LabelEncoder to all non-numeric columns
        if df_obj.shape[1] > 0:
            df_obj.apply(lambda x: self.encoder[x.name].fit(x))
        return self

    def transform (self, X):
        df_obj = X[X.columns[X.dtypes==object]]
        if df_obj.shape[1] < 1:
            return X
        result = df_obj.apply(lambda x: self.encoder[x.name].transform(x))
        # add the unchanged numeric columns to the result
        df_num = X[X.columns[X.dtypes!=object]]
        for col in df_num.columns:
            result[col] = df_num[col]
        return result









