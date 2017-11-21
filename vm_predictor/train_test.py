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

from crome_multi import CromeProcessor

filename = "thirty-two.csv"
features = ['day', 'weekday', 'hour', 'minute', 'hist-1D8H', 'hist-1D4H', 'hist-1D2H', 'hist-1D1H', 'hist-1D', 'hist-1D15m', 'hist-1D30m', 'hist-1D45m']

cp = CromeProcessor ('cpu_usage', feats=features)
model = cp.build_model_from_CSV(filename)

print ("predict_CSV")
predictions = cp.predict_CSV(filename, resample="15min", data_out="test_multi.csv")
print (predictions[:5])

print ("model prediction")
import pandas as pd
df = pd.read_csv("test_multi.csv")
preds = model.predict(df[features])
print (preds[:5])
