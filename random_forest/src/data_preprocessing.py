import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)
    print(data.head())
    print(data.info())
    print(data.describe())
    X = data.drop(columns=['Daily_Revenue'])
    y = data['Daily_Revenue']
    return X, y

X,y = load_data('data/coffee_shop_revenue.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)



