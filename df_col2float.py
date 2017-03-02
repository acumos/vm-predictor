
import sys
import pandas as pd
import numpy as np
import os.path

input_file = sys.argv[1]
output_file = sys.argv[2]
columns = sys.argv[3:]

df = pd.read_csv(input_file)
for colname in columns:
    print "Processing column: ", colname
    df[colname] = df[colname].apply(lambda x:np.float64(str(x).replace(",","")))

df.to_csv(output_file, index=False, compression='gzip' if os.path.splitext(output_file)[1]=='.gz' else None)    
