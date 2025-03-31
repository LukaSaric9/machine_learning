import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import os
import json
import joblib

def load_data():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    return X_train,y_train

def train_models():
    X_train, y_train = load_data() 

    os.makedirs('models', exist_ok=True)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    for name,model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{name.lower()}.pkl')
        print(f'Trained and saved {name}')

if __name__ == '__main__':
    train_models()

