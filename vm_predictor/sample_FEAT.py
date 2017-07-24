from __future__ import print_function

import pandas as pd
from os.path import isfile, join, basename, exists
from random import sample



def read_and_prep (fname, entity_col, work_col):
    print ("read file", fname)
    df = pd.read_csv(fname)
    print ("  prep column ", entity_col)
    for colname in df.columns:
        new_name = colname.replace(" ", "")
        if new_name == entity_col:
            df[work_col] = df[colname]                    # keep the original column!
            df = trim_columns (df, [work_col])
            return df
    return None
    
    

def trim_columns (df, columns):
    for colname in columns:
        print ("  trim: ", colname)
        try:
            df[colname] = df[colname].str.strip()
        except:
            pass
    return df


        



if __name__ == "__main__":

    print ("main")
    import argparse
    
    
    entity_col = "VM_ID"
    
    parser = argparse.ArgumentParser(description = "CROME training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    
    parser.add_argument('files', nargs='+', help='list of CSV files to process')
    parser.add_argument('-p', '--percent', help = 'percent of total to sample', type=float, default=5.0)
    parser.add_argument('-o', '--output_file', help = 'destination output file', default='sample_out.csv')

    cfg = parser.parse_args()
    working_col = entity_col + "_working"

    
    if not exists (cfg.output_file):
        print ("PASS ONE")
        
        all_entities = set()
        
        for fname in cfg.files:
            df = read_and_prep (fname, entity_col, working_col)
            all_entities = all_entities | set(df[working_col])
            
        to_keep = int(len(all_entities) * cfg.percent / 100.0)
        print ("found total of %s entities, will keep %s" % (len(all_entities), to_keep))
        
        winners = sample (all_entities, to_keep)            # note:  winners is a LIST
        
        
        print ("PASS TWO")                                  # collect all rows of the sampled VMs
        final_df = pd.DataFrame()
        
        for fname in cfg.files:
            df = read_and_prep (fname, entity_col, working_col)
            
            print ("  filter")
            df = df[df[working_col].isin(winners)]
        
            print ("    rows extracted: ", len(df))
            final_df = pd.concat([final_df, df])
    
        df = pd.DataFrame()     # release memory
        print ("Total rows in sampled dataframe:  ", len(final_df))
        print ("cleanup")
        final_df.drop (working_col, axis=1, inplace=True)
        print ("Writing ", cfg.output_file)
        final_df.to_csv (cfg.output_file, index=False)
        
        
    
    else:
        print ("output file exists:  ", cfg.output_file)


    

