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
    df[colname] = df[colname].str.strip()
	
print ("writing")    
df.to_csv(output_file, index=False)    
