#Imports
import requests
import json
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

#SKLearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def train_evaluate(df, features, label, model_list):
    """
    Train and evaluate machine learning models for a given list of algorithms.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset
    - features (list): List of feature names
    - label (String): The label
    - model_list (list): List of machine learning algorithms (pre-instantiated models)

    Returns:
    - results_df (pd.DataFrame): DataFrame containing algorithm names and corresponding mse and r2 scores
    """
    results = []
    
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    for algorithm in model_list:
        model = algorithm
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        results.append({'Algorithm': algorithm.__class__.__name__, 'Mean Squared Error': mse, 'Algorithm': algorithm.__class__.__name__, 'R2': r2})

    results_df = pd.DataFrame(results)
    return results_df

def hyperparameter_opti(df, features, label, model_list):
    """
    Train and evaluate machine learning models for a given list of algorithms.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset
    - features (list): List of feature names
    - label (String): The label
    - model_list (list): List of machine learning algorithms (pre-instantiated models)

    Returns:
    - results_df (pd.DataFrame): DataFrame containing algorithm names and corresponding mse and r2 scores
    """
    results = []
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    for algorithm, params in model_list:
        # Perform GridSearchCV for hyperparameter tuning
        gs_best = GridSearchCV(algorithm, params, cv=5, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = algorithm.set_params(**gs_best.best_params_)
        # Fit the final model on the entire training set
        final_model.fit(X_train, y_train)
        prediction = final_model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        results.append({'Algorithm': algorithm.__class__.__name__, 'Mean Squared Error': mse, 'Algorithm': algorithm.__class__.__name__, 'R2': r2})
        print({'Algorithm': algorithm.__class__.__name__, 'Mean Squared Error': mse, 'Algorithm': algorithm.__class__.__name__, 'R2': r2})
              
    results_df = pd.DataFrame(results)
    return results_df



repositories = []
parameters={ 'q': 'stars:>=50', 'per_page': 100,  'page': 1   }
while len(repositories) < 1000:
  response = requests.get('https://api.github.com/search/repositories', params=parameters)
  if response.status_code == 200:
    batch = response.json()['items']
    repositories.extend(batch)
    print("Batch size: " ,len(batch), " Total number of repositories: ", len(repositories))
    if 'next' in response.links:
      parameters['page'] += 1
    else:
      break
print(repositories[:10])

fieldnames = repositories[0].keys()
print(fieldnames)
with open("original_project_data.csv", 'w', newline='', encoding='utf-8') as csv_file:
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  writer.writeheader()
  for repo in repositories:
    writer.writerow(repo)

data = pd.read_csv("original_project_data.csv")
data.drop_duplicates()

print(data.info())
data = data.drop(columns=[col for col in data.columns if 'url' in col])
data = data.drop(columns=[col for col in data.columns if data[col].dtype == 'object'])
data = data.replace({True: 1, False: 0})
data = data.drop(columns=['id', 'private', 'fork', 'disabled', 'allow_forking', 'watchers', 'forks', 'score', "watchers_count", "open_issues_count"])

correlation_matrix_2 = data.corr()

plt.figure(figsize=(20, 10))
heatmap = plt.imshow(correlation_matrix_2, cmap = 'cubehelix_r', interpolation = 'nearest')

for i in range(len(correlation_matrix_2)):
    for j in range(len(correlation_matrix_2)):
        plt.text(j, i, f"{correlation_matrix_2.iloc[i, j]:.2f}", ha='center', va='center', color='r')


plt.xticks(range(len(correlation_matrix_2)), correlation_matrix_2.columns, rotation=45)
plt.yticks(range(len(correlation_matrix_2)), correlation_matrix_2.columns)
plt.title("Correlation Matrix")
plt.colorbar(heatmap)
plt.show()

label = 'stargazers_count'
features = data.columns.to_list()
features.remove('stargazers_count')
print("Label:", label)
print("Features:", features)

X = data[features]
y = data[label]

model_list =[RandomForestRegressor(), DecisionTreeRegressor(), LinearRegression(), SVR(), KNeighborsRegressor()]

train_evaluate(data, features, label, model_list)

knn_params = {"n_neighbors": range(2, 10)}
dt_params = {'max_depth': range(1, 5),
             "min_samples_split": range(2, 10)}
rf_params = {"max_depth": [5, 20, None],
             "n_estimators": [150, 300]}
lr_params = {"n_jobs" : range(1, 5)}
sv_params = {'C': [0.1, 1, 10, 100],
             'gamma': ['scale', 'auto', 0.1, 1, 10]}

regressors = [(RandomForestRegressor(), rf_params),
              (DecisionTreeRegressor(), dt_params),
              (LinearRegression(), lr_params),
              (SVR(), sv_params),
              ]