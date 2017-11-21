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

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator


h2o.init()
h2o.h2o.no_progress()

def H2O_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    if verbose:
        print (">> Building model for target ", target_col)

    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    if verbose:
        print (">>   importing:", train_path)
    train_frame = h2o.import_file(path=train_path)

    if verbose:
        print (">>   importing:", test_path)
    test_frame = h2o.import_file(path=test_path)

    if verbose:
        print (">>   training...")
    res = rf_model.train (x=feat_cols, y=target_col, training_frame=train_frame)

    if verbose:
        print (">>   predicting...")
    preds = rf_model.predict(test_frame)

    predicted = preds.as_data_frame()

    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)
    h2o.remove(rf_model)

    return predicted['predict'].values
