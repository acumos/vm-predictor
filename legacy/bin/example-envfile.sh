#!/usr/bin/env bash
# ===============LICENSE_START=======================================================
# Acumos Apache-2.0
# ===================================================================================
# Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
# ===================================================================================
# This Acumos software file is distributed by AT&T and Tech Mahindra
# under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============LICENSE_END=========================================================
rm -f example-python.war

curl -X POST \
--form pojo=@GBM_model_python_1463864606917_1.java \
--form jar=@h2o-genmodel.jar \
--form python=@score.py \
--form pythonextra=@vectorizer.pickle \
--form pythonextra=@lib/modelling.py \
--form pythonextra=@lib/__init__.py \
--form envfile=@sp2.yaml \
localhost:55000/makepythonwar > example-python.war

if [ -s example-python.war ]
then
  echo "Created example-python.war"
  echo "Run with run-example-python.sh"
else
  echo "Failed to build example.war"
  exit 1
fi

