# Wrapping Models for Deployment
To utilize this transformer model set, it first creates a detect transformer
and then a pixelate transformer.
Continue to the [next tutorial](lesson2.md)
to see how to utilize these models with a simple demo API server.



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


## Data Processing

Given a set of FEAT (or other) CSV files containing time-series data, the process is fairly simple.

1. First one needs to decide whether to build multi-entity (VM) models or single-entity models.
2. Single-entity simulations process files that contain time-series data for
   a single entity (VM) only.   Since the FEAT csv files typically contain multiple
   entities, they must first be broken up into per-entity files using the
   tool `util/add_FEAT_data.py`.

   It may be as simple as this:
```
    python vm_predictor/util/add_FEAT_data.py FEAT*.csv -o ./VM_data
```

   Then processing the separate VM files, with compound charts, could be accomplished by:
```
    python vm_predictor/crome_multi.py  ./VM_data/*.csv -c
```

3.  Multi-entity simulations can process the FEAT files directly.  However, some care must be taken
    if the files are sequential or very large.  If the FEAT files are
    sequential in time, you do not want to process them separately;  instead
    you want to process them as if they were concatenated.  That can be
    accomplished with the `join_files` (-j) option:

```
    python vm_predictor/crome_multi.py -j FEAT*.csv
```

4. Depending on memory constraints, you may not be able to process all of the
   concatenated FEAT files at once.  What you can do instead is process 1 or 2
   at a time and collect the results in intermediate JSON files using the
   `write_predictions` (-p) option.

   For example, if the train-predict regimen is 31 days and 1 day, at MINIMUM
   two month-long files are required.   So to cover a longer time span one
   could proceed in steps as follows:
```
    python vm_predictor/crome_multi.py -p -o ./predict -j FEAT_Feb2017.csv FEAT_Mar2017.csv
    python vm_predictor/crome_multi.py -p -o ./predict -j FEAT_Mar2017.csv FEAT_Apr2017.csv
    python vm_predictor/crome_multi.py -p -o ./predict -j FEAT_Apr2017.csv FEAT_May2017.csv
```

5. The final output of that will be a set of JSON prediction files, one per
   entity/VM covering the entire (4-month) time range. To create charts from
   those JSON files another tool is used:  *preds2charts.py*.   For example:
```
    python vm_predictor/preds2charts.py ./predict/*.json
```



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

crome_multi.py and its helper file preds2charts.py build charts displaying
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


## Model Deployment

To run the model training and push to a respective back-end server, use the installed
script ``run_vm-predictor_reference.py``.  As a convenience, to run the script locally
_without installing_ (during development), use the commmand ``bin/run_local.sh``.

This repo currently includes example training and testing data.  You can create a model
and push it to a locally running Acumos mock server with the following example.

**NOTE** The examples pushing to a library are using the reference `testing/upload/app.py`
server in the main `acumos` package to simulate backend testing in these examples.


* multiple VM training - where the *wisdom of the crowd* can be utilized and
  multiple VMs (that have not been seen before) can be predicted upon. This
  is possible by learning general patterns from multiple VMs (not specific to one
  identity) and applying those patterns to other VMs

```
# training + dump for a multi-vm model in a directory (raw data)
python vm_predictor/crome_multi.py -t cpu_usage -o data/multi_feature -f day weekday hour minute hist_1D8H hist_1D4H hist_1D2H hist_1D1H hist_1D hist_1D15m hist_1D30m hist_1D45m VM_ID -c -P 2 -d model_multi data/multi/raw-feature.csv.gz

# training + push for a multi-vm model in a directory (raw data) -- note, asssumes localhost testing server, user:foo, pass:bar
python vm_predictor/crome_multi.py -t cpu_usage -o data/multi_feature -f day weekday hour minute hist_1D8H hist_1D4H hist_1D2H hist_1D1H hist_1D hist_1D15m hist_1D30m hist_1D45m VM_ID -c -P 2 data/multi/raw-feature.csv.gz -a "http://localhost:8887/v2/upload" -A "http://localhost:8887/v2/auth"
```

* single VM training, preprocessed data - an example where a model is trained
  for a single VM.  These models may be higher performance than the multi-model
  version, but they are smaller and more sensitive to training times.

```
# training + dump for a single model in a directory (raw data)
python vm_predictor/crome.py -t cpu_usage -f day weekday hour minute hist_1D VM_ID -d model_single data/single/train.csv

# training + push for a single model in a directory (raw data) -- note, asssumes localhost testing server, user:foo, pass:bar
python vm_predictor/crome.py -t cpu_usage -f day weekday hour minute hist_1D VM_ID -d model_single data/single/train.csv -a "http://localhost:8887/v2/upload" -A "http://localhost:8887/v2/auth"
```


### Advanced multi-VM training examples

This example trains and predicts multi-VM models on dates in February and March
with target net_usage, outputs Simple and Compound charts to folder "FebMar",
and uses a combination of datetime and historical features.

```
python vm_predictor/crome_multi.py -j ff/FEAT_VM_1702_1703.csv ff/FEAT_VM_1703_1704.csv -s -c -t net_usage -f day weekday hour minute hist_1D8H hist_1D4H hist_1D2H hist_1D1H hist_1D hist_1D15m hist_1D30m hist_1D45m -o ./FebMar
```

This example trains and predicts multi-VM models through December and January
on 'cpu_usage' (the default), but uses only the FIRST FIFTY entities (VMs)
found in the files.  Predictions are not charted but are written as JSON files
to folder './json'.  Also, the VM_ID column is added as a feature.

```
python vm_predictor/crome_multi.py -j ff/FEAT_VM_1612_1701.csv ff/FEAT_VM_1701_1702.csv -p -o ./json -v 50 -f day weekday hour minute hist_1D8H hist_1D4H hist_1D2H hist_1D1H hist_1D hist_1D15m hist_1D30m hist_1D45m VM_ID
```

This example trains and predicts multi-VM models on target 'net_usage' using
only the first 10 VMs in the file FEAT_sampled.csv.  The prediction interval
is 7 days.  Compound charts are output.  The ML model "Extra Trees With Scaling"
is selected and the number of trees is set to 5.

```
python vm_predictor/crome_multi.py FEAT_sampled.csv -c -t net_usage -v 10 -o sk_test_et_sc -i et__n_estimators 5 -P 7 -f day weekday hour minute hist_1D8H hist_1D4H hist_1D2H hist_1D1H hist_1D hist_1D15m hist_1D30m hist_1D45m VM_ID -M ET_SC
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

