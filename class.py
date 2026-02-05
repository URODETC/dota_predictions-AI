import joblib
from prediction_model import PredictionModel

import __main__

__main__.PredictionModel = PredictionModel
gg = joblib.load("model/prediction.pkl")
print(gg.prediction([1, 2, 3, 4, 5], [5, 6, 7, 8, 9]))
