# Computing Steps 1-2
# Test Fit making a 75%-25% split

import numpy as np
import pandas as pd
import random
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from verstack.stratified_continuous_split import scsplit

# Set a Random State value
RANDOM_STATE = 7
# Set Python random a fixed value
random.seed(RANDOM_STATE)
# Set numpy random a fixed value
np.random.seed(RANDOM_STATE)

#Load dataset
filename = 'experiments_code/executionset_regression_1.csv'
names = ['management', 'soil_type.CL', 'soil_type.LS', 'soil_type.SL', 'precipitation', 'relative_irrigation', 'number_of_trips_reduction', 'relative_profit_percentage']
dataset = pd.read_csv(filename, delim_whitespace=True, header=0)


# Split-out validation dataset
num_of_dataset_training_columns = 6
array = dataset.values
X =  dataset.iloc[:,0:num_of_dataset_training_columns]


Y =  dataset.iloc[:,num_of_dataset_training_columns]
validation_size = 0.25

num_folds = 10

# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('Bayes Ridge', BayesianRidge()))
models.append(('Ridge Regression', RidgeCV()))
models.append(('LASSO', Lasso(random_state=RANDOM_STATE)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=RANDOM_STATE)))
models.append(('SVMR', SVR()))



time_start_step1 = time.perf_counter()

# For Step 1 -- Simple splitting
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state = RANDOM_STATE)

print('\n')
print('Simple Random Splitting')
print('------------------------------------------------------')
print('\n')


r2_fit_scores=[]
r2_predict_scores=[]
RMSE_predict_scores=[]


for name, model in models:   
    fit = model.fit(X_train,Y_train)
    r2_fit_scores.append(fit.score(X_train, Y_train))
    prediction = model.predict(X_validation)    
    # Calculate the R squared -- it is the accuracy for regression
    r2_predict_scores.append(metrics.r2_score(Y_validation,prediction))
    RMSE_predict_scores.append(metrics.mean_squared_error(Y_validation,prediction))

max_r2_predict_score=max(r2_predict_scores)
min_rmse_predict_score=min(RMSE_predict_scores)

print('R2 Train Scores (Optimal is value 1)\n')  
for score in r2_fit_scores:
   print(models[r2_fit_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('------------------------------------------------------')

print('R2 Test Scores (Optimal is value 1)\n')  
for score in r2_predict_scores:
    print(models[r2_predict_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('------------------------------------------------------')

print('RMSE Scores (Optimal is value 0)\n')    
for score in RMSE_predict_scores:
    print(models[RMSE_predict_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('\n')
print('Best R2 Score\n',models[r2_predict_scores.index(max_r2_predict_score)][0], ':', round(max_r2_predict_score, 5), '\n')  

print('Best RMSE Score\n',models[RMSE_predict_scores.index(min_rmse_predict_score)][0], ':', round(min_rmse_predict_score, 5), '\n')

time_end_step1 = time.perf_counter()

time_start_step2 = time.perf_counter()

# For Step 2 -- Startified splitting
X_train, X_validation, Y_train, Y_validation = scsplit(X, Y, stratify = Y,
                                         test_size = 0.25, random_state = RANDOM_STATE)

print('\n')
print('Stratified Splitting')
print('------------------------------------------------------')
print('\n')

r2_fit_scores=[]
r2_predict_scores=[]
RMSE_predict_scores=[]


for name, model in models:   
    fit = model.fit(X_train,Y_train)
    r2_fit_scores.append(fit.score(X_train, Y_train))
    prediction = model.predict(X_validation)    
    # Calculate the R squared -- it is the accuracy for regression
    r2_predict_scores.append(metrics.r2_score(Y_validation,prediction))
    RMSE_predict_scores.append(metrics.mean_squared_error(Y_validation,prediction))

max_r2_predict_score=max(r2_predict_scores)
min_rmse_predict_score=min(RMSE_predict_scores)

print('R2 Train Scores (Optimal is value 1)\n')  
for score in r2_fit_scores:
   print(models[r2_fit_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('------------------------------------------------------')

print('R2 Test Scores (Optimal is value 1)\n')  
for score in r2_predict_scores:
    print(models[r2_predict_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('------------------------------------------------------')

print('RMSE Scores (Optimal is value 0)\n')    
for score in RMSE_predict_scores:
    print(models[RMSE_predict_scores.index(score)][0], ':', round(score, 5)) #*100,'%')

print('\n')
print('Best R2 Score\n',models[r2_predict_scores.index(max_r2_predict_score)][0], ':', round(max_r2_predict_score, 5), '\n')  

print('Best RMSE Score\n',models[RMSE_predict_scores.index(min_rmse_predict_score)][0], ':', round(min_rmse_predict_score, 5), '\n')  

time_end_step2 = time.perf_counter()


print(f"Execution duration of step 1: {time_end_step1 - time_start_step1:0.4f} seconds")
print(f"Execution duration of step 2: {time_end_step2 - time_start_step2:0.4f} seconds")
