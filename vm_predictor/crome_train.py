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

# This is run within the predictor
import os
import h2o
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator

from textblob import TextBlob

# Shared user library between training and scoring
from lib.modelling import saveModel

# Should be input parameter
MODELS_DESTINATION_DIR = "./lib"


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words if len(word) > 0 and word.isalpha() ]

# Load data
def load_data(filename):
     # Note: skiping wrong rows
     return np.genfromtxt(filename, dtype=None, delimiter='\t', names=['label', 'message'], skip_header=0, invalid_raise=False)

def tf_idf(corpus):
    vectorizer = TfidfVectorizer(
            #analyzer = 'word',
            analyzer = split_into_lemmas,
            stop_words = 'english',
            min_df=0,
            decode_error = 'ignore',
            strip_accents = 'ascii',
            ngram_range=(1,3))
    # Fit and transform input corpus
    model = vectorizer.fit_transform(corpus)
    return (vectorizer, model)


# this won't work:  TOO SLOW
def calc_avg_abs_err (predicted, actual):
    total = 0.0
    count = 0
    for row in range(predicted.nrows):
        try:
            val = actual[row, 0]
        except Exception, e:
            print "ERROR: ", e
        if val == val:                  # NAN test
            num = abs(predicted[row,0] - val)
            den = max(predicted[row,0], val)
            if den == 0:
                aae = 0.0
            else:
                aae = num / den
            total += aae
            count += 1
            if count % 10000 == 0:
                print count
    mean_aae = total / count
    return mean_aae


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



def train_random_forest(file_path, target_col):
    print "Building model..."

    h2o.init()
    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    print "  importing", file_path
    mainframe = h2o.import_file(path=file_path)
    #mainframe = h2o.import_file(path="http://vision5.research.att.com:8001/vnf_bandwidth_timecols.csv")
    #train_frame, validate_frame, test_frame = mainframe.split_frame([0.60, 0.20])
    train_frame, test_frame = mainframe.split_frame([0.50])

    cols = [u'SUBSCRIBER_NAME', u'month', u'day', u'weekday', u'hour', u'minute']
    print "  training..."
    #res = rf_model.train (x=cols, y=target_col, training_frame=train_frame, validation_frame=validate_frame)
    res = rf_model.train (x=cols, y=target_col, training_frame=train_frame)

    print "  predicting..."
    preds = rf_model.predict(test_frame)

    # predictions are in preds
    print "  calculating AAE..."
    #aae = calc_avg_abs_err (preds, test_frame[target_col])
    aae = calc_AAE (preds, test_frame, target_col)
    print "AAE=", aae
    print "done!"


#
# Main entry point. Accepts parameters
#  for example:
#    ipython train.py -- --verbose --models-dir /tmp/models
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CROME model builder & tester")
    parser.add_argument('--datafile', help = 'Input data file', type=str, dest='file', default = 'http://localhost:8001/vnf_bandwidth_timecols.csv')
    parser.add_argument('--models-dir', help = 'Directory to save generated models', type=str, default = MODELS_DESTINATION_DIR)
    parser.add_argument('--verbose', help = 'More detailed output', dest='verbose', action='store_true')
    parser.add_argument('--target', help = 'Target column name', type=str, dest='target', default = 'usage')

    cfg = parser.parse_args()
    train_random_forest(cfg.file, cfg.target)

