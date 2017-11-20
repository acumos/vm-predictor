# Advanced Machine Learning Topics
This tutotial discusses more background information for the problem of
temporal pattern prediction as it applied to this model and its data.
Deployment and testing information can be found in [the previous lesson](lesson2.md).

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

