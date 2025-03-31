import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_excel('data/ENB2012_data.xlsx')
print(data.head())
print(data.info())
print(data.describe())

#standardization 
scaler = StandardScaler()   
data[['X1','X2','X3','X4','Y1','Y2']] = scaler.fit_transform(data[['X1','X2','X3','X4','Y1','Y2']])

print(data)

#separate the data into X and y
X = data.drop(columns=['Y1','Y2'])
y = data[['Y1','Y2']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

