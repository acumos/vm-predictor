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

import sys
import pandas as pd
import numpy as np
import os.path

input_file = sys.argv[1]
output_file = sys.argv[2]
columns = sys.argv[3:]

print ("reading")
df = pd.read_csv(input_file)
for colname in columns:
    print ("processing: ", colname)
    df[colname] = df[colname].apply(lambda x:np.float64(str(x).replace(",","")))

print ("writing")
df.to_csv(output_file, index=False, compression='gzip' if os.path.splitext(output_file)[1]=='.gz' else None)
