# vm-predictor
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



## DATA

Data files can be large, and CROME's so-called 'FEAT' files are no exception.
(Note:   Another set of tools captures the raw data to create the FEAT files.
The workings of those tools are beyond the scope of this document.)

The FEAT data may be delimited by month.   For example, `data/multi/raw-feature.csv.gz`
contains data from February 2017 through the beginning of March 2017.  The data is
"raw" in that it needs a substantial amount of preprocessing.   That's obvious just
by looking at the original column names:

```
    [' cpu        _ usage       ', ' cpu        _ usagemhz    ', ' mem        _ active      ', ' mem        _ consumed    ', ' mem        _ granted     ', ' mem        _ usage       ', ' net        _ received    ', ' net        _ transmitted ', ' net        _ usage       ', 'DATETIMEUTC', 'DATEUTC', 'GLOBAL_CUSTOMER_ID', 'SUBSCRIBER_NAME', 'Unnamed: 0', 'VM_ID']
```

And many of the rows may require additional formatting, such as removing
padding before and after strings.

The FEAT files are typically over 1 GB in size.  That makes processing them
unwieldy, especially when running simulations which span a large segment of
time (i.e. many months).


## TOOLS

The workhorse script for processing the FEAT (e.g.) files is *crome_multi.py*.
It does everything from reformatting columns to training models to building charts.

But the essence of crome_multi.py is to implement a train/predict regimen on
a given file using the concept of a "sliding window".

Basically crome_multi.py sees the incoming data as a single time series,
sorted from earliest date to latest;  in fact it pre-processes incoming CSV
files to conform to that view.  It executes the indicated train/predict regimen
across the entire time series by:  (A) training a model using a specified
training duration, e.g. 31 days;  and (B) using that same model to predict
data for the specified prediction duration, e.g. 1 day, which
*immediately follows* the training data.  This window is then advanced one
prediction period, and the cycle repeats until the data is exhausted.

For example, given data from January to December and current defaults of a
31-day training period and a 1-day prediction period, crome_multi.py will
first build a model using data from January 1st through January 31st, and make
predictions for February 1st.  That gives one day's worth of data.  Then it
advances one day (the prediction period) and builds a new model with data from
January 2nd to February 1st, and predicts February 2nd.  That's the second
day's data.  And so on, until predictions are made through December 31st.

The results for the entire time period are saved as charts and/or prediction
data.   See below for more details.


## SAMPLING

Ordinarily the rows in the CSV data files represent data sampled at a certain
frequency, such as every 5 minutes.  crome_multi.py has the ability to
resample this to another value, averaging all the values in that interval.
The default setting of *15 minutes* may be changed using the '-S' command
line option.   (Obviously you can only use values that are a multiple of the
original sample interval.)   Note that this value is enforced early during
preprocessing, and all subseqent computations use the re-sampled values.


## CHARTS

`crome_multi.py` and its helper file `preds2charts.py` build charts displaying
the predicted target value vs. the actual value through the entire time
range contained in the input file(s).

There are six different individual charts (-s option), and one compound
chart (-c option) currently available.

Individual charts are:

* "Original":  displays the original data in the working sample size (-S option).
* "Percentile_95":  displays the daily 95th percentile of the target value.
* "STD":  displays the daily standard deviation of the target.
* "Variance":  displays the daily variance of the target.
* "Busy_Hour_4H":  displays the daily busiest hour from 0-23 (4-hour window).
* "Busy_Avg_4H":  displays the daily mean of the target during the busy hour.

As written, the -s option, when present, will write all 6 chart types, but
could easily be enhanced to select specific ones.

The compound chart is simply all 6 simple charts displayed on one page.


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


## Example Usages
Please consult the [tutorials](tutorials) dirctory for usage examples
including [training examples](tutorials/lesson1.md)
and an in-place [web page demonstration](tutorials/lesson2.md).

## Release Notes
The [release notes](release-notes.md) catalog additions and modifications
over various version changes.


## Additional Background
Some additional information is provided in [advanced backgrounds tutorial](tutorials/lession3.md)
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
