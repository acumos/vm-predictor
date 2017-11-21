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
import numpy as np
from os.path import exists, join


def remove_column_spaces (df):
    replace_dict = {}
    for colname in df.columns:
        replace_dict[colname] = colname.replace(" ", "")

    df = df.rename(index=str, columns=replace_dict)
    return df



def cols_to_float (df, columns):
    for colname in columns:
        print ("  float: ", colname)
        try:
            df[colname] = df[colname].apply(lambda x:np.float64(str(x).replace(",","")))
        except:
            pass
    return df


def trim_columns (df, columns):
    for colname in columns:
        print ("  trim: ", colname)
        try:
            df[colname] = df[colname].str.strip()
        except:
            pass
    return df



def main(input_filename, dest_dir, float_cols, trim_cols, max_proc, skip_existing, VM_list=[]):
    print ("read file", input_filename)
    df = pd.read_csv(input_filename)

    print ("remove spaces")
    df = remove_column_spaces(df)

    print ("collect VMs")
    to_trim = list(trim_cols)
    df = trim_columns (df, ['VM_ID'])
    to_trim.remove('VM_ID')

    if not VM_list or len(VM_list) == 0:
        VM_list = sorted(list(set(df['VM_ID'])))            # process all VMs
    else:
        df = df[df['VM_ID'].isin(VM_list)]

    print ("convert columns to float:", float_cols)
    df = cols_to_float (df, float_cols)

    print ("apply trim:", to_trim)
    df = trim_columns (df, to_trim)

    for vm in VM_list[:max_proc]:
        df_vm = df[df['VM_ID']==vm]
        print ("processing VM: %s (%s rows)" % (vm, len(df_vm)))
        output_filename = join(dest_dir, vm + ".csv")
        if exists (output_filename):
            if skip_existing:
                print ("  already exists")
                continue
            df_orig = pd.read_csv(output_filename)
            df_vm = pd.concat([df_orig, df_vm])
        big_size = len(df_vm)
        df_vm = df_vm.drop_duplicates(subset="DATETIMEUTC")
        if len(df_vm) != big_size:
            print ("  dropped %s duplicate rows" % (big_size - len(df_vm),))
        df_vm.to_csv(output_filename, index=False)

    print ("done")




if __name__ == "__main__":
    import argparse

    float_cols = ["cpu_usage", "cpu_usagemhz", "mem_active", "mem_consumed", "mem_granted", "mem_usage", "net_received",  "net_transmitted", "net_usage"]
    trim_cols = ["DATETIMEUTC", "DATEUTC", "GLOBAL_CUSTOMER_ID", "SUBSCRIBER_NAME", "VM_ID"]

    parser = argparse.ArgumentParser(description = "CROME data import tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_files', nargs='+', help='name of FEAT_VM*.csv file(s) to import')
    parser.add_argument('-o', '--output_dir', default='.', help='destination directory containing VM data')
    parser.add_argument('-v', '--vm_list', nargs='+', help='list of VMs to process (default=all)')
    parser.add_argument('-n', '--max_vm', help = 'process at most N entities per input file', type=int, default=1000000)
    parser.add_argument('-k', '--skip_existing', help = 'do not modify existing output files', action='store_false')

    cfg = parser.parse_args()

    for feat_file in cfg.input_files:
        main (feat_file, cfg.output_dir, float_cols, trim_cols, cfg.max_vm, cfg.skip_existing, VM_list=cfg.vm_list)

