import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_test_data():
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_test, y_test

def evaluate_models():
    X_test, y_test = load_test_data()

    os.makedirs('metrics', exist_ok=True)

    metrics = {}

    for model_name in ['LinearRegression', 'Ridge', 'Lasso']:
        model = joblib.load(f'models/{model_name.lower()}.pkl')
        y_pred = model.predict(X_test)

        metrics[model_name] = {}
        for i,target in enumerate(['Y1', 'Y2']):
            metrics[model_name][target] = {
                'MSE': mean_squared_error(y_test[target], y_pred[:, i]),
                'R2': r2_score(y_test[target], y_pred[:, i]),
            }

    with open(f'metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Metrics saved")
    return metrics

if __name__ == '__main__':
    evaluate_models()