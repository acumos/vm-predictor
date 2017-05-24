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

    

def main(input_filename, dest_dir, float_cols, trim_cols, VM_list=[]):
    print ("read file", input_filename)

    df = pd.read_csv(input_filename)

    print ("remove spaces")
    df = remove_column_spaces(df)
    
    print ("collect VMs")
    df = trim_columns (df, ['VM_ID'])
    trim_cols.remove('VM_ID')
    
    if not VM_list or len(VM_list) == 0:
        VM_list = list(set(df['VM_ID']))            # process all VMs
    else:
        df = df[df['VM_ID'].isin(VM_list)]
    
    print ("convert columns to float:", float_cols)
    df = cols_to_float (df, float_cols)
    
    print ("apply trim:", trim_cols)
    df = trim_columns (df, trim_cols)

    for vm in VM_list:
        df_vm = df[df['VM_ID']==vm]
        print ("processing VM: %s (%s rows)" % (vm, len(df_vm)))
        output_filename = join(dest_dir, vm + ".csv")
        if exists (output_filename):
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
    parser.add_argument('input_file', help='name of FEAT_VM*.csv file to import')
    parser.add_argument('output_dir', help='destination directory containing VM data')
    parser.add_argument('-v', '--vm_list', nargs='+', help='list of VMs to process (default=all)')
    
    cfg = parser.parse_args()
    
    #main (cfg.input_file, cfg.output_dir, float_cols, trim_cols, VM_list=["0625561c-8305-44be-afea-6b2bd6d3cdb0"])
    #main (cfg.input_file, cfg.output_dir, float_cols, trim_cols, VM_list=["0c1babd4-9ff2-4891-8da8-908d83c9c758"])
    main (cfg.input_file, cfg.output_dir, float_cols, trim_cols, VM_list=cfg.vm_list)
    
 
    







