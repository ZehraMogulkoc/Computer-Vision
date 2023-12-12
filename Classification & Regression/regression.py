import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso


df = pd.read_csv("D:\\Computer-Vision\\Classification & Regression\\cars.csv")

# Convert 'Origin' to numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['Origin'], drop_first=True)

X = df.drop('MPG', axis=1)
y = df['MPG']

# Handle missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Drop rows with missing values in the target variable
missing_target_rows = y.isnull()
X = X[~missing_target_rows]
y = y.dropna()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressors = {
    'Linear Regression': LinearRegression(),
    'Robust Regression': RANSACRegressor(),
    'Lasso Regression': Lasso(alpha=0.1, 
              precompute=True, 
              positive=True, 
              selection='random',)
}

results_table = pd.DataFrame(columns=['Method', 'MAE', 'MSE', 'R^2'])

for name, regressor in regressors.items():
    scores_mae = -cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    scores_mse = -cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores_r2 = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')
    results_table = results_table.append({
        'Method': name,
        'MAE': scores_mae.mean(),
        'MSE': scores_mse.mean(),
        'R^2': scores_r2.mean()
    }, ignore_index=True) 


for method, regressor in regressors.items():
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_table = results_table.append({
        'Method': method + ' (Test)',
        'MAE': mae,
        'MSE': mse,
        'R^2': r2
    }, ignore_index=True)


print(results_table)





