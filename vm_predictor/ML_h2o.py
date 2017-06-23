
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

    
h2o.init()
h2o.h2o.no_progress()    
    
def H2O_train_and_predict(train_path, test_path, target_col, feat_cols, verbose=False):
    if verbose:
        print (">> Building model for target ", target_col)

    rf_model = H2ORandomForestEstimator (response_column=target_col, ntrees=20)
    if verbose:
        print (">>   importing:", train_path)
    train_frame = h2o.import_file(path=train_path)

    if verbose:
        print (">>   importing:", test_path)
    test_frame = h2o.import_file(path=test_path)

    if verbose:
        print (">>   training...")
    res = rf_model.train (x=feat_cols, y=target_col, training_frame=train_frame)

    if verbose:
        print (">>   predicting...")
    preds = rf_model.predict(test_frame)

    predicted = preds.as_data_frame()

    h2o.remove(train_frame.frame_id)
    h2o.remove(test_frame.frame_id)
    h2o.remove(preds.frame_id)
    h2o.remove(rf_model)

    return predicted['predict'].values
