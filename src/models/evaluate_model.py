import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import json

os.chdir(os.path.dirname(__file__))
X_test_scaled = pd.read_csv('../../data/processed_data/X_test_scaled.csv', index_col=0)
y_test = pd.read_csv('../../data/processed_data/y_test.csv', index_col=0)
y_test = y_test.iloc[:, 0]

with open("../../models/model.pkl", "rb") as file:
    rf = pickle.load(file)

print('Evaluating the model...')

metric = {'R2': rf.score(X_test_scaled, y_test)}

with open('../../metrics/scores.json', 'w') as file:
    json.dump(metric, file)

y_pred = rf.predict(X_test_scaled)
pd.DataFrame(y_pred).to_csv('../../data/y_test_pred.csv')
