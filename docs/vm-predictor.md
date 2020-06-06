<!---
.. ===============LICENSE_START=======================================================
.. Acumos CC-BY-4.0
.. ===================================================================================
.. Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
.. ===================================================================================
.. This Acumos documentation file is distributed by AT&T and Tech Mahindra
.. under the Creative Commons Attribution 4.0 International License (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://creativecommons.org/licenses/by/4.0
..
.. This file is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ===============LICENSE_END=========================================================
-->

# VM Predictor Guide
An example model that can predict resource utilization at a given timestamp
based on the time and historical context.


## INTRODUCTION

Time series data is often cyclical in nature, with ups and downs dependent on
days of the week, holidays, morning vs afternoon, or whatever.  The main idea
behind this project is to predict future values of a time series, as
represented by rows in a CSV file, using machine learning (ML) techniques.

As an example, the CROME data files contain VM (virtual machine) usage
statistics at 5-minute intervals, including cpu, memory, network, and others.
The hope is that by predicting future VM behavior we can more efficiently allocate resources.

This set of tools allows the user to simulate (and, potentially, implement) a
train/predict regimen over an extended period of time.   It can help set up
scenarios such as:  "let's train on 31 days of data and predict usage for the
following day, repeated every day for 6 months".


## WORKFLOW

Given a set of FEAT (or other) CSV files containing time-series data, the process is fairly simple.

1. Collect KPI data from a collector software. A simple columnar format is acceptable, with
   the minum columns being *KPI*, *timestamp*, *VM/uniqueid*.
2. Next, decide if a single-entity or multi-entitiy model is preferred.  Generally, we
   advocate for a multi- model because some insights from one VM may be conveyed to another.
3. Either reformat data by hand, use the scripts provided here, or let it occur during
   training and processing.
    1. Reformatting requires the collection of all time samples and resampling at the
       specified intervals.
    2. Specifically, resampling history for 1D8H means that using the provided sample
       time one should also include a sample from one day and eight hours ago.
4. Train the model and export a binary artifact.  Note that these models are currently
   static and not online-updatable.


## MACHINE LEARNING BASICS

Machine learning models are trained on "features" in order to predict the "target".

* In CROME FEAT files the target is typically a column containing one of the
  usage statistics:  cpu_usage, mem_usage, or net_usage.   The target is
  specified on the command line with the `-t` option, e.g. `-t net_usage`.
  Note that target defaults to `cpu_usage`.
* The features used are, by default, only time-based features such as 'month',
  'day', 'weekday', 'hour', and 'minute'.   These do not require any other
  information in the CSV file other than the date.  Good performance can be
  achieved using just those features.
* For enhanced ML performance however additional features may be required.
  When the default is not used ALL features must be listed on the command line
  with the "-f" switch.
* In some cases the data files themselves contain features of value.  Just add
  the name of the desired column to the feature list, for example "VM_ID".
  Additionally crome_multi.py provides specialized syntax to give access to
  prior values of the target, as features.  If a feature begins with "hist_"
  it indicates such a 'historical' feature.  The time displacement string
  immediately follows, for example 'hist_1D' is the target value one day previous.
  'hist_1H' is one hour previous;  'hist_1D1H' is one day plus one hour previous.  And so on.
* Those historical values are point values (according to the base sample size)
  so to sample over a longer period add a second parameter after a dash.
  'hist_2D-1H' specifies the previous target value two days ago averaged over one hour.

See the [training and deployment steps](docs/tutorials/lession1.md) for more examples.


## MODELS

Access to several ML model types are built in to crome_multi.py.   The `-M`
command line option allows selecting the learning algorithm (model).
Current choices include:

* "RF" -- Random Forest (default)
* "RF_SC" -- Random Forest with Scaler
* "ET" -- Extra Trees
* "ET_SC" -- Extra Trees with Scaler

The set_param (-i) switch gives command-line access to one or more of the
model's internal parameters.   If "RF" is selected (the default), one can
for example set the number of estimators to 18 with:  `-i rf__n_estimators 18`.

Code for choices "H2O" and "ARIMA" also exists but require a scikit
wrapper to function within crome_multi.py (not included).
Also, the base class can easily accomodate your own *custom* models
especially via the scikit interface.


### Installation and Package dependencies

To install vm-predictor just clone this repository and use pip.

Note You must have installed acumos_client before this package can be installed.

```
git clone --depth 1 <vm-predictor repo url>
pip install .
```

Package dependencies for the core code and testing have been flattened into a single file
for convenience. Instead of installing this package into your your local
environment, execute the command below.
```
pip install -r requirments.txt
```
or, if you want ot install dependencies with a classic package place holder...
```
pip install . -v
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
  -d DUMP_MODEL, --dump_model DUMP_MODEL
                        dump model to a directory for local running
                        (default: )

  (only for single-model mode)
  -R, --is_raw_data     for the push and dump options, perform feature
                        processing (default: False)
```




### Advanced Variations

As you may have guessed by now there are a lot of choices in ML models and
their parameters.   One way to zero in on "best practice" is to do a Grid
Search.    The basic idea is that all the various options and their values
form a grid of possibilities, and to find the ideal choice we try all the
combinations.  The script *variations.py* demonstrates one way of doing that.
Essentially each "run_variation" line instantiates a different scikit model
with assorted parameters.  Though not an exhaustive search (each run is lengthy),
it is meant as example code which, when you get the hang of it, will serve as
a gateway to your own experiments.


## Example Usages
Please consult the [tutorials](tutorials) dirctory for usage examples
including [training examples](tutorials/lesson1.md)
and an in-place [web page demonstration](tutorials/lesson2.md).

## Release Notes
The [release notes](release-notes.md) catalog additions and modifications
over various version changes.


## Additional Background
Some additional information is provided in [advanced backgrounds tutorial](tutorials/lesson3.md)
for those readers with interest in general machine learning study for this problem.


### FILE REFERENCE

File | Description
-----|------------
*add_FEAT_data.py*  | Extracts individual VM data from one or more FEAT data files.  Use '-h' for a list of options.
*crome.py*  | Older version of crome_multi.py, can only process single-VM models.  Includes H2O and ARIMA support.
*crome_multi.py*    | Main CROME processing script builds multi-VM models and simulates train-predict cycles over an extended time period, outputting the results as charts or tables.   Please consult the help page (-h) for a complete list of options.
*df_col2float.py*   |  Convert a column to floating point.
*df_colnospc.py*  |  Remove spaces from column names.
*df_column.py* | Display set of all values in a given column for a CSV file.
*df_cols.py* | Show names of columns in a dataframe.
*df_concat.py* | Concatenate dataframe CSVs with same column layout.
*df_head.py* |  Display the first few rows of a CSV file.
*df_sample.py* | Subsample a CSV data file.  Set 'preserve_ones' to keep EULR=1 rows.
*df_shape.py* | Display shape of CSV file(s) (rows, columns).
*df_split.py* | Randomly split a dataframe into 2 separate files by a given percentage (default=0.5).
*df_tail.py* |  Display the last few rows of a CSV file.
*df_trim.py* |  Remove leading and trailing blanks from string values.
*ML_arima.py* | Plug-in component for the ARIMA model.  Not available in crome_multi.
*ML_h2o.py* | Plug-in component for using H2O models.  Not available in crome_multi.
*preds2charts.py*  |  Builds charts from prediction JSON files.   See the help page (-h) for additional options.
*sample_FEAT.py* |  A utility script which allows taking a random sample of the VMs in FEAT*.csv data.
*showFiles.py* | This tool launches a little web server allowing viewing of local charts and other files via a web browser.
*StringColumnEncoder.py* | Encode a dataframe's string columns as part of a pipeline.
*train_test.py* | Example code demonstrating training and testing from CSV files.
*variations.py* | Utility program which can run a "grid search" of model variations to find the best parameters.  For most authoratative results should be used with a FEAT file containing a random sampling of VMs.  Meant as a starting point for further experiments.  Note:  long run time.

# Metadata Examples
* [example catalog image](catalog_image.png)

