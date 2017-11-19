#!/usr/bin/env bash

echo "starting prediction service on port 55002"

java -jar ../jetty-runner-8.1.14.v20131031.jar --port 55002 example-python.war 

