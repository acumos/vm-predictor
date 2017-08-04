from __future__ import print_function



import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor        
from sklearn.ensemble import ExtraTreesRegressor        

from crome import CromeProcessor

float_cols = ["cpu_usage", "cpu_usagemhz", "mem_active", "mem_consumed", "mem_granted", "mem_usage", "net_received",  "net_transmitted", "net_usage"]
trim_cols = ["DATETIMEUTC", "DATEUTC", "GLOBAL_CUSTOMER_ID", "SUBSCRIBER_NAME", "VM_ID"]
features = ['VM_mapped', 'day', 'weekday', 'hour', 'minute', 'hist-1D8H', 'hist-1D4H', 'hist-1D2H', 'hist-1D1H', 'hist-1D', 'hist-1D15m', 'hist-1D30m', 'hist-1D45m']


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

    
    
    

def preprocess (df, target_col, VM_list=[], max_proc=50):
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
    
    vm_map = dict([(val, i) for i, val in enumerate(set(df['VM_ID']))])
    df['VM_mapped'] = df['VM_ID'].apply(lambda x: vm_map[x])
    
    print ("convert columns to float:", float_cols)
    df = cols_to_float (df, float_cols)
    print ("apply trim:", trim_cols)
    df = trim_columns (df, trim_cols)

    cp = CromeProcessor (target_col, feats=features)
    
    result = pd.DataFrame()
    
    print ("processing VM list...")
    for vm in VM_list[:max_proc]:
        #df_vm = df[df['VM_ID']==vm].copy()
        df_vm = df[df['VM_mapped']==vm_map[vm]]
        
        print ("  VM: %s (%s rows)" % (vm, len(df_vm)))
        df_vm = cp.transform_dataframe (df_vm)
        
        # keep only feature columns + target + VM
        df_vm = df_vm[features + [target_col]]
        result = pd.concat ([result, df_vm])
        
        # remove this VM from the original dataframe to speed things up
        df = df[df['VM_mapped'] != vm_map[vm]]
        
    print ("finished.  result has %s rows." % len(result))
    return result, vm_map



def build_model_from_file (filename, target_col):
    print ("reading", filename)
    df = pd.read_csv(filename)
    df, mapper = preprocess (df, target_col)
    df.to_csv("training_features.csv")
    print ("training")
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(df[features], df[target_col])
    return {"model":rf, "mapper":mapper}

    
if __name__ == "__main__":
    thing = build_model_from_file ("FEAT_VM_1612_1701.csv", 'cpu_usage')
    #thing = build_model_from_file ("FEAT_sampled.csv", 'cpu_usage')
    import pickle
    print ("pickling")
    pickle.dump(thing, open("model.pkl", "wb"))
    
   
    