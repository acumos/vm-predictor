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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from os import path


def test_training(monkeypatch):
    env_update(monkeypatch)
    import sys
    print(sys.path)

    # run imports after injecting local path
    from vm_predictor.crome_multi import CromeProcessor
    from vm_predictor.StringColumnEncoder import StringColumnEncoder

    pathRoot = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    filename = path.join(pathRoot, 'data', 'multi', "raw-feature.csv.gz")
    features = ['day', 'weekday', 'hour', 'minute', 'hist_1D8H', 'hist_1D4H', 'hist_1D2H', 'hist_1D1H', 'hist_1D', 'hist_1D15m', 'hist_1D30m', 'hist_1D45m']
    ML_model = Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20))])

    cp = CromeProcessor('cpu_usage', feats=features, model=ML_model)

    print("predict_CSV")
    results, VM_list = cp.process_CSVfiles([filename])
    print(results[:5])

    # TODO: generate prediction method
    # print ("model prediction")
    # import pandas as pd
    # df = pd.read_csv("data/multi/test-features.csv")
    # preds = ML_model.predict(df[features])
    # print (preds[:5])
    pass


def env_update(monkeypatch):
    import sys
    pathRoot = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    print("Adding '{:}' to sys path".format(pathRoot))
    if pathRoot not in sys.path:
        monkeypatch.syspath_prepend(pathRoot)
