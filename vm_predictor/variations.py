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

from __future__ import print_function

from .crome_multi import CromeProcessor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from vm_predictor.StringColumnEncoder import StringColumnEncoder

import numpy as np


def run_variation(var_name, cp):
    input_file = "big_sample.csv"
    result_file = "results_var.txt"

    print ("")
    print ("RUNNING VARIATION:  ", var_name)
    result, VM_list = cp.process_CSVfiles([input_file])

    sum_error = 0.0
    total_weight = 0
    errors = []
    for vm in VM_list:
        df = result[result[cp.entity_col]==vm]
        if len(df) > 0:
            err = cp.calc_AAE (df)
            print ("    score for %s = %s" % (vm, err))
            if err >= 0.99:
                import pdb; pdb.set_trace()
            if not np.isnan(err):
                errors.append(err)
                weight = len(df)
                sum_error += weight * err
                total_weight += weight

    overall_error = sum_error / total_weight
    print ("OVERALL ERROR: ", overall_error)

    open(result_file, 'a+').write ("%s, %s\n" % (var_name, overall_error))
    return overall_error



def variations (target):
    ft = ['day', 'weekday', 'hour', 'minute', 'hist-1D8H', 'hist-1D4H', 'hist-1D2H', 'hist-1D1H', 'hist-1D', 'hist-1D15m', 'hist-1D30m', 'hist-1D45m']
    sh = ['day', 'weekday', 'hour', 'minute', 'hist-1H', 'hist-4H', 'hist-1D', 'hist-15m']
    run_variation (target+" RF_20 std_features", CromeProcessor(target, feats=ft, model=RandomForestRegressor(n_estimators=20)))
    run_variation (target+" SC_RF_20 std_features", CromeProcessor(target, feats=ft, model=Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20))])))
    run_variation (target+" ET_20 std_features", CromeProcessor(target, feats=ft, model=ExtraTreesRegressor(n_estimators=20)))
    run_variation (target+" RF_10 std_features", CromeProcessor(target, feats=ft, model=RandomForestRegressor(n_estimators=10)))
    run_variation (target+" RF_20 vm_features", CromeProcessor(target, feats=ft+['VM_ID'], model=Pipeline([('enc', StringColumnEncoder()), ('rf', RandomForestRegressor(n_estimators=20))])))
    run_variation (target+" RF_20 sh_features", CromeProcessor(target, feats=sh, model=RandomForestRegressor(n_estimators=20)))
    run_variation (target+" SC_ET_10 vm_features", CromeProcessor(target, feats=ft+['VM_ID'], model=Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('et', ExtraTreesRegressor(n_estimators=10))])))
    run_variation (target+" RF_100 std_features", CromeProcessor(target, feats=ft, model=RandomForestRegressor(n_estimators=100)))
    run_variation (target+" SC_RF_20 vm_features", CromeProcessor(target, feats=ft+['VM_ID'], model=Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20))])))
    run_variation (target+" SC_ET_5 vm_features", CromeProcessor(target, feats=ft+['VM_ID'], model=Pipeline([('enc', StringColumnEncoder()), ('sc', StandardScaler()), ('et', ExtraTreesRegressor(n_estimators=5))])))



if __name__ == "__main__":
    import sys
    target = sys.argv[1]
    variations(target)


