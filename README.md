# vm-predictor

An example model that can predict resource utilization at a given timestamp based on the time and historical context.

# Installation

To install vm-predictor just clone this repository and use pip.  

**Note** You must have installed `cognita-python-client` before this package can be installed.  It is not included in the requirements file at this time because it is stored on a private repository, likely at [this url](../cognita-python-client/src).
```
git clone <vm-predictor repo url>
pip install .
```

To run the model training and push to a respective back-end server:
```
run_vm-predictor_reference.py
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

# Testing

TBD

