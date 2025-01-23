import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

os.chdir(os.path.dirname(__file__))
X_train_scaled = pd.read_csv('../../data/processed_data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('../../data/processed_data/y_train.csv', index_col=0)
y_train = y_train.iloc[:, 0]

with open("../../models/params.pkl", "rb") as file:
    params = pickle.load(file)

print('Training the model...')

rf = RandomForestRegressor(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
rf.fit(X_train_scaled, y_train)

with open('../../models/model.pkl', "wb") as file:
    pickle.dump(rf, file)
