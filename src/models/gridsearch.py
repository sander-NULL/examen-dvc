import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle

os.chdir(os.path.dirname(__file__))
X_train_scaled = pd.read_csv('../../data/processed_data/normalized/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('../../data/processed_data/split/y_train.csv', index_col=0)
y_train = y_train.iloc[:, 0]

print('Doing grid search and finding best hyperparameters...')

rf = RandomForestRegressor()
params = {'n_estimators': [5,10,15,25,50,75,100],
          'max_depth': range(1,6)}

grid_search = GridSearchCV(rf, params, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

with open('../../models/params.pkl', "wb") as file:
    pickle.dump(grid_search.best_params_, file)