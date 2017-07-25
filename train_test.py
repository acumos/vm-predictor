from crome_multi import CromeProcessor

filename = "thirty-two.csv"
features = ['day', 'weekday', 'hour', 'minute', 'hist-1D8H', 'hist-1D4H', 'hist-1D2H', 'hist-1D1H', 'hist-1D', 'hist-1D15m', 'hist-1D30m', 'hist-1D45m']

cp = CromeProcessor ('cpu_usage', feats=features)
model = cp.build_model_from_CSV(filename)

print ("predict_CSV")
predictions = cp.predict_CSV(filename, resample="15min", data_out="test_multi.csv")
print (predictions[:5])

print ("model prediction")
import pandas as pd
df = pd.read_csv("test_multi.csv")
preds = model.predict(df[features])
print (preds[:5])
