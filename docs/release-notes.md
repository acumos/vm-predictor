# Release Notes
## 0.8
### 0.8.2
* Update requirements for matplotlib

### 0.8.1
* Documentation and package update to use install instructions instead of installing
  this package directly into a user's environment.
* Refactor documentation into sections and tutorials.
* Create this release notes document for better version understanding.
* Refactor to remote the demo `bin` scripts and rewire for direct call of the
  script `crome.py` and `crome_multi.py` as the primary interaction mechanisms.

### 0.8.0
* Refactor for compliant dataframe usage following primary client library
  examples for repeated columns (e.g. dataframes) instead of custom types
  that parsed rows individually.
* Refactor web, api, main model wrapper code for corresponding changes.
* Migration from previous library structure to new acumos client library
* Refactor to not need **this** library as a runtime/installed dependency

