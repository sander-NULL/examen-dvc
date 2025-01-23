import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(__file__))
df = pd.read_csv('../../data/raw_data/raw.csv', index_col=0)

X = df.drop(columns='silica_concentrate')
y = df['silica_concentrate']

print('Splitting data into train and test sets...')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.to_csv('../../data/processed_data/X_train.csv')
X_test.to_csv('../../data/processed_data/X_test.csv')
y_train.to_csv('../../data/processed_data/y_train.csv')
y_test.to_csv('../../data/processed_data/y_test.csv')