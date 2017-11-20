# Wrapping Models for Deployment
To utilize this transformer model set, it first creates a detect transformer
and then a pixelate transformer.
Continue to the [next tutorial](lesson2.md)
to see how to utilize these models with a simple demo API server.



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
python vm_predictor/crome.py -t cpu_usage -d single_model -f day weekday hour minute hist_1D VM_ID -d model_single data/single/train.csv

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

