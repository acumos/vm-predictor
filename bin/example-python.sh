#!/usr/bin/env bash

rm -f example-python.war

curl -X POST \
--form pojo=@DRF_model_python_1484947403759_39.java \
--form jar=@h2o-genmodel.jar \
--form python=@/home/mtinnemeier/raw_input_ezlog2b.py \
localhost:55000/makepythonwar > example-python.war

if [ -s example-python.war ]
then
  echo "Created example-python.war"
  echo "Run with run-example-python.sh"
else
  echo "Failed to build example.war"
  exit 1
fi

