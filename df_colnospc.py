from __future__ import print_function

import sys
import pandas as pd
import numpy as np
import os.path

input_file = sys.argv[1]
output_file = sys.argv[2]



df = pd.read_csv(input_file)


replace_dict = {}


for colname in df.columns:
	replace_dict[colname] = colname.replace(" ", "")
	
df = df.rename(index=str, columns=replace_dict)
df.to_csv(output_file, index=False)    
