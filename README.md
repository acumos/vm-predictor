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
./bin/run_local.sh -t cpu_usage -a "http://localhost:8887/v1/models" data/train.csv
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
  -a PUSH_ADDRESS, --push_address PUSH_ADDRESS
                        server address to push the model (default: )
```



# Testing

A simple unit test is planned but not currently available.

