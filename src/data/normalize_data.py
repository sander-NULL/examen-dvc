import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.chdir(os.path.dirname(__file__))
X_train = pd.read_csv('../../data/processed_data/split/X_train.csv', index_col=0)
X_test = pd.read_csv('../../data/processed_data/split/X_test.csv', index_col=0)

print('Normalizing train and test sets...')

scaler = MinMaxScaler()

X_train_scaled = X_train.copy()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

X_test_scaled = X_test.copy()
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

X_train_scaled.to_csv('../../data/processed_data//normalized/X_train_scaled.csv')
X_test_scaled.to_csv('../../data/processed_data/normalized/X_test_scaled.csv')