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


print ("starting")

import pandas as pd

print ("reading")

#input_filename= "FEAT_sampled_nospc_tcols_flt_trim.csv"
input_filename= "BIG_Dec16Jan17_nospc_tcols_flt_trim.csv"

df = pd.read_csv(input_filename)

print ("date conversion")
date_col = "DATETIMEUTC"
DT = 'DT'
df[DT] = pd.to_datetime(df[date_col])       # WHO does this conversion and WHEN ?

print ("sorting")
df = df.sort_values(DT)

print ("collect VMs")
ss = list(set(df['VM_ID']))

dest = "./VM_ID/"


for vm in ss[:100]:
    print ("write VM: ", vm)
    df_vm = df[df['VM_ID']==vm]
    output_filename = dest + vm + ".csv"
    df_vm.to_csv(output_filename, index=False)

print ("done")

