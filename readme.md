__CROME Project Files__
-----------------------


__INTRODUCTION__

Time series data is often cyclical in nature, with ups and downs dependent on days of the week, holidays, morning vs afternoon, or whatever.  The main idea behind this project is to predict future values of a time series, as represented by rows in a CSV file, using machine learning (ML) techniques.

As an example, the CROME data files contain VM (virtual machine) usage statistics at 5-minute intervals, including cpu, memory, network, and others.  The hope is that by predicting future VM behavior we can more efficiently allocate resources.

This set of tools allows the user to simulate (and, potentially, implement) a train/predict regimen over an extended period of time.   It can help set up scenarios such as:  "let's train on 31 days of data and predict usage for the following day, repeated every day for 6 months".



__DATA__

Data files can be large, and CROME's so-called 'FEAT' files are no exception.   (Note:   Another set of tools captures the raw data to create the FEAT files.  The workings of those tools are beyond the scope of this document.)

The FEAT data may be delimited by month.   For example, "FEAT_VM_1702_1703.csv" contains data from February 2017 through the beginning of March 2017.  The data is "raw" in that it needs a substantial amount of preprocessing.   That's obvious just by looking at the original column names:

    [' cpu        _ usage       ', ' cpu        _ usagemhz    ', ' mem        _ active      ', ' mem        _ consumed    ', ' mem        _ granted     ', ' mem        _ usage       ', ' net        _ received    ', ' net        _ transmitted ', ' net        _ usage       ', 'DATETIMEUTC', 'DATEUTC', 'GLOBAL_CUSTOMER_ID', 'SUBSCRIBER_NAME', 'Unnamed: 0', 'VM_ID']

And many of the rows may require additional formatting, such as removing padding before and after strings.

The FEAT files are typically over 1 GB in size.  That makes processing them unwieldy, especially when running simulations which span a large segment of time (i.e. many months).


__TOOLS__

The workhorse script for processing the FEAT (e.g.) files is *crome_multi.py*.   It does everything from reformatting columns to training models to building charts.

But the essence of crome_multi.py is to implement a train/predict regimen on a given file using the concept of a "sliding window".

Basically crome_multi.py sees the incoming data as a single time series, sorted from earliest date to latest;  in fact it pre-processes incoming CSV files to conform to that view.  It executes the indicated train/predict regimen across the entire time series by:  (A) training a model using a specified training duration, e.g. 31 days;  and (B) using that same model to predict data for the specified prediction duration, e.g. 1 day, which *immediately follows* the training data.  This window is then advanced one prediction period, and the cycle repeats until the data is exhausted.

For example, given data from January to December and current defaults of a 31-day training period and a 1-day prediction period, crome_multi.py will first build a model using data from January 1st through January 31st, and make predictions for February 1st.  That gives one day's worth of data.  Then it advances one day (the prediction period) and builds a new model with data from January 2nd to February 1st, and predicts February 2nd.  That's the second day's data.  And so on, until predictions are made through December 31st.

The results for the entire time period are saved as charts and/or prediction data.   See below for more details.



__SAMPLING__

Ordinarily the rows in the CSV data files represent data sampled at a certain frequency, such as every 5 minutes.  crome_multi.py has the ability to resample this to another value, averaging all the values in that interval.   The default setting of *15 minutes* may be changed using the '-S' command line option.   (Obviously you can only use values that are a multiple of the original sample interval.)   Note that this value is enforced early during preprocessing, and all subseqent computations use the re-sampled values.  



__CHARTS__

crome_multi.py and its helper file preds2charts.py build charts displaying the predicted target value vs. the actual value through the entire time range contained in the input file(s).     

There are six different individual charts (-s option), and one compound chart (-c option) currently available.

Individual charts are:

    "Original":  displays the original data in the working sample size (-S option).
    "Percentile_95":  displays the daily 95th percentile of the target value.
    "STD":  displays the daily standard deviation of the target.
    "Variance":  displays the daily variance of the target.
    "Busy_Hour_4H":  displays the daily busiest hour from 0-23 (4-hour window).
    "Busy_Avg_4H":  displays the daily mean of the target during the busy hour.
    
As written, the -s option, when present, will write all 6 chart types, but could easily be enhanced to select specific ones.    
    
The compound chart is simply all 6 simple charts displayed on one page. 


__WORKFLOW__

Given a set of FEAT (or other) CSV files containing time-series data, the process is fairly simple.

(1) First one needs to decide whether to build multi-entity (VM) models or single-entity models.   

(2) Single-entity simulations process files that contain time-series data for a single entity (VM) only.   Since the FEAT csv files typically contain multiple entities, they must first be broken up into per-entity files using the tool *add_FEAT_data.py*.   

It may be as simple as this:

    python add_FEAT_data.py FEAT*.csv -o ./VM_data

Then processing the separate VM files could be accomplished by:

    python crome_multi.py  ./VM_data/*.csv


(3) Multi-entity simulations can process the FEAT files directly.  However, some care must be taken if the files are sequential or very large.

If the FEAT files are sequential in time, you do not want to process them separately;  instead you want to process them as if they were concatenated.  That can be accomplished with the 'join_files' (-j) option:

    python crome_multi.py -j FEAT*.csv
    
Depending on memory constraints, you may not be able to process all of the concatenated FEAT files at once.  What you can do instead is process 1 or 2 at a time and collect the results in intermediate JSON files using the write_predictions (-p) option.

For exaple, if the train-predict regimen is 31 days and 1 day, at MINIMUM two month-long files are required.   So to cover a longer time span one could proceed in steps as follows:

    python crome_multi.py -p -o ./predict -j FEAT_Feb2017.csv FEAT_Mar2017.csv
    python crome_multi.py -p -o ./predict -j FEAT_Mar2017.csv FEAT_Apr2017.csv
    python crome_multi.py -p -o ./predict -j FEAT_Apr2017.csv FEAT_May2017.csv
    Etc.
    
The final output of that will be a set of JSON prediction files, one per entity/VM covering the entire (4-month) time range.
To create charts from those JSON files another tool is used:  *preds2charts.py*.   For example:

    python preds2charts.py ./predict/*.json
    




__MACHINE LEARNING BASICS__

Machine learning models are trained on "features" in order to predict the "target".  

In CROME FEAT files the target is typically a column containing one of the usage statistics:  cpu_usage, mem_usage, or net_usage.   The target is specified on the command line with the '-t' option, e.g. "-t net_usage".   Note that target defaults to 'cpu_usage'.

The features used are, by default, only time-based features such as 'month', 'day', 'weekday', 'hour', and 'minute'.   These do not require any other information in the CSV file other than the date.  Good performance can be achieved using just those features.

For enhanced ML performance however additional features may be required.   

When the default is not used ALL features must be listed on the command line with the "-f" switch.  

In some cases the data files themselves contain features of value.  Just add the name of the desired column to the feature list, for example "VM_ID".

Additionally crome_multi.py provides specialized syntax to give access to prior values of the target, as features.  If a feature begins with "hist-" it indicates such a 'historical' feature.  The time displacement string immediately follows, for example 'hist-1D' is the target value one day previous.  'hist-1H' is one hour previous;  'hist-1D1H' is one day plus one hour previous.  And so on.

Those historical values are point values (according to the base sample size) so to sample over a longer period add a second parameter after a dash.  'hist-2D-1H' specifies the previous target value two days ago averaged over one hour.


See below for more examples.


__MODELS__

Access to several ML model types are built in to crome_multi.py.   

The -M command line option allows selecting the learning algorithm (model).   Current choices include:

    "RF" -- Random Forest (default)
    "RF_SC" -- Random Forest with Scaler
    "ET" -- Extra Trees
    "ET_SC" -- Extra Trees with Scaler

The set_param (-i) switch gives command-line access to one or more if the model's internal parameters.   If "RF" is selected (the default), one can for example set the number of estimators to 18 with:  "-i rf__n_estimators 18".

Code for choices "H2O" and "ARIMA" also exists but require a scikit wrapper to function within crome_multi.py (not included).

Also, the base class can easily accomodate your own *custom* models especially via the scikit interface.

   

__EXAMPLES__

This example trains and predicts multi-VM models on dates in February and March with target net_usage, outputs Simple and Compound charts to folder "FebMar", and uses a combination of datetime and historical features.

    python crome_multi.py -j ff/FEAT_VM_1702_1703.csv ff/FEAT_VM_1703_1704.csv -s -c -t net_usage -f day weekday hour minute hist-1D8H hist-1D4H hist-1D2H hist-1D1H hist-1D hist-1D15m hist-1D30m hist-1D45m -o ./FebMar

   
This example trains and predicts multi-VM models through December and January on 'cpu_usage' (the default), but uses only the FIRST FIFTY entities (VMs) found in the files.  Predictions are not charted but are written as JSON files to folder './json'.  Also, the VM_ID column is added as a feature.

    python crome_multi.py -j ff/FEAT_VM_1612_1701.csv ff/FEAT_VM_1701_1702.csv -p -o ./json -v 50 -f day weekday hour minute hist-1D8H hist-1D4H hist-1D2H hist-1D1H hist-1D hist-1D15m hist-1D30m hist-1D45m VM_ID


This example trains and predicts multi-VM models on target 'net_usage' using only the first 10 VMs in the file FEAT_sampled.csv.  The prediction interval is 7 days.  Compound charts are output.  The ML model "Extra Trees With Scaling" is selected and the number of trees is set to 5.

    python crome_multi.py FEAT_sampled.csv -c -t net_usage -v 10 -o sk_test_et_sc -i et__n_estimators 5 -P 7 -f day weekday hour minute hist-1D8H hist-1D4H hist-1D2H hist-1D1H hist-1D hist-1D15m hist-1D30m hist-1D45m VM_ID -M ET_SC



    
__FILE REFERENCE__    



File | Description
-----|------------
*add_FEAT_data.py*  | Extracts individual VM data from one or more FEAT data files.  Use '-h' for a list of options.
*crome_multi.py*    | Main CROME processing script builds multi-VM models and simulates train-predict cycles over an extended time period, outputting the results as charts or tables.   Please consult the help page (-h) for a complete list of options.
*preds2charts.py*  |  Builds charts from prediction JSON files.   See the help page (-h) for additional options.
*df_cols.py* | Show names of columns in a dataframe.
*df_column.py* | Display set of all values in a given column for a CSV file.
*df_concat.py* | Concatenate dataframe CSVs with same column layout.
*df_sample.py* | Subsample a CSV data file.  Set 'preserve_ones' to keep EULR=1 rows.
*df_shape.py* | Display shape of CSV file(s) (rows, columns).
*df_split.py* | Randomly split a dataframe into 2 separate files by a given percentage (default=0.5).
*df_col2float.py*   |  Convert a column to floating point.
*df_colnospc.py*  |  Remove spaces from column names.
*df_head.py* |  Display the first few rows of a CSV file.
*df_tail.py* |  Display the last few rows of a CSV file.
*df_trim.py* |  
*push_cognita.py* |  Experimental code to push a model to the Cognita platform.
*StringColumnEncoder.py* | Encode a dataframe's string columns as part of a pipeline.
*crome.py*  | Older version of crome_multi.py, can only process single-VM models.  Includes H2O and ARIMA support.
*ML_h2o.py* | Plug-in component for using H2O models.  Not available in crome_multi.
*ML_arima.py* | Plug-in component for the ARIMA model.  Not available in crome_multi.
*showFiles.py* | This tool launches a little web server allowing viewing of local charts and other files via a web browser.



