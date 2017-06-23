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

