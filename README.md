# vm-predictor

An example model that can predict resource utilization at a given timestamp based on the time and historical context.

# Installation

To install vm-predictor just clone this repository and use pip.  

**Note** You must have installed `cognita-python-client` before this package can be installed.  It is not included in the requirements file at this time because it is stored on a private repository, likely at [this url](../cognita-python-client/src).
```
git clone <vm-predictor repo url>
pip install .
```

## Live training

To run the model training and push to a respective back-end server, use the installed 
script ``run_vm-predictor_reference.py``.  As a convenience, to run the script locally 
without installing (during development), use the commmand ``bin/run_local.sh``.

This repo currently includes example training and testing data.  You can create a model
and push it to a locally running Cognita mock server with the following example.

```
# training + dump for a single model in a directory (raw data)
./bin/run_local.sh data/multi/raw-feature.csv.gz -t cpu_usage -o data/multi_feature -f day weekday hour minute hist-1D8H hist-1D4H hist-1D2H hist-1D1H hist-1D hist-1D15m hist-1D30m hist-1D45m VM_ID -c -P 2 -d model -R data/multi/raw-feature.csv.gz

# training + push to a running server (preprocessed data)
./bin/run_local.sh -t cpu_usage -a "http://localhost:8887/v1/models" data/single/train.csv

# training + dump for a single model in a directory (preprocessed data)
./bin/run_local.sh -t cpu_usage -d single_model -f day weekday hour minute hist-1D VM_ID -d model data/single/train.csv
```

### Grid search

One feature of the multi-VM code is to allow a grid search of a few different
parameters.  Generally, this requires **raw** features as 
input so that the various can be utilized in the feature aggregation process.
Note: There are still more paramters that can be tuned, but this script
explores those with the biggest potential gains in performance. *(added 7/23)*
```
# train on raw features, produce performance plots across variants
TBD
```


## Live evaluation

To test the model on a swagger instance, you'll need to find the running instance and open its swagger page in a browser. (**NOTE** These instructions may change with the evolution of the back-end server) 

* First find the running instance by probing the running instances.  Look for a recent model, or dereference with the model named 'vm-predictor'
```
<server url>/solutions/running
```
* Then open your corresponding page in a browser
```
<server url>/swagger:8123
```

## General runtime arguments

The main script has these commandline arguments, which can also be evoked with the option `-h`.

```
VM Predictor training and testing

positional arguments:
  files                 list of CSV files to process

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        target prediction column (default: cpu_usage)
  -c, --compound        output compound charts (default: False)
  -s, --separate        output separate charts (default: False)
  -r, --randomize       randomize file list (default: False)
  -p PNG_DIR, --png_dir PNG_DIR
                        destination directory for PNG files (default: )
  -n MAX_FILES, --max_files MAX_FILES
                        process at most N files (default: 1000000)
  -m MIN_TRAIN, --min_train MIN_TRAIN
                        minimum # samples in a training set (default: 300)
  -D DATE_COL, --date_col DATE_COL
                        column to use for datetime index (default:
                        DATETIMEUTC)
  -T TRAIN_DAYS, --train_days TRAIN_DAYS
                        size of training set in days (default: 31)
  -P PREDICT_DAYS, --predict_days PREDICT_DAYS
                        number of days to predict per iteration (default: 1)
  -S SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        desired duration of train/predict units. See
                        http://pandas.pydata.org/pandas-
                        docs/stable/timeseries.html#offset-aliases (default:
                        15min)
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
                        list of features to use (default: ['day', 'weekday',
                        'hour', 'minute'])
  -M ML_PLATFORM, --ML_platform ML_PLATFORM
                        specify machine learning platform to use (default: SK)
  -R, --is_raw_data     for the push and dump options, perform feature
                        processing (default: False)
  -a PUSH_ADDRESS, --push_address PUSH_ADDRESS
                        server address to push the model (default: )
  -d DUMP_PICKLE, --dump_pickle DUMP_PICKLE
                        dump model to a pickle directory for local running
                        (default: )
```



# Testing

A simple unit test is planned but not currently available.

