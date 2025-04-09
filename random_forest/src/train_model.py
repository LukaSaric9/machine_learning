import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze() #pass it as 1d arrray instead of dataframe
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze() 
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators = 100, random_state=42,n_jobs=-1)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")

    return pred,r2, mse, mae

def save_model(model, filename='models/random_forest_model.pkl'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  
    joblib.dump(model, filename)
    print(f"🧠 Model saved to {filename}") 

def plot_feature_importance(model, X_train):
    feature_importances = model.feature_importances_

    feature_names = X_train.columns
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
    plt.title('Feature Importance')
    plt.show()

def plot_predicted_vs_actual(y_test, pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, c=y_test, cmap='viridis', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of equality
    plt.xlabel('Actual Daily Revenue')
    plt.ylabel('Predicted Daily Revenue')
    plt.title('Predicted vs Actual Daily Revenue')
    plt.colorbar(label='Actual Daily Revenue')
    plt.show()

def main():
    print("🚀Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    print("🛠️Training model...")
    model = train_model(X_train, y_train)

    print("🔍Evaluating model...")
    pred,r2,mse,mae = evaluate_model(model,X_test,y_test)

    print("💾Saving model...")
    save_model(model)

    print("📊 Plotting Feature Importance...")
    plot_feature_importance(model, X_train)

    print("📊 Plotting Predicted vs Actual...")
    plot_predicted_vs_actual(y_test, pred)

if __name__ == "__main__":
    main()