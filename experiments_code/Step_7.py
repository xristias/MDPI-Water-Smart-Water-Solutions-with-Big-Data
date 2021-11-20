# Computing Step 7
# Use uncorrelated data with proper transformation and the optimal repeated cv parameters
# after sensitivity analysis
# for Hyperparameter Tuning (SVMR was the prevailing algorithm) 

import numpy as np
from numpy.lib.function_base import average, diff
import pandas as pd
import random
import time	
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
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

num_of_dataset_training_columns = 6
array = dataset.values
X =  dataset.iloc[:,0:num_of_dataset_training_columns]
Y =  dataset.iloc[:,num_of_dataset_training_columns]
validation_size = 0.25
# Test options and evaluation metric
num_folds = 10
seed = 7

#stratified
X_train, X_validation, Y_train, Y_validation = scsplit(X, Y, stratify = Y,
                                         test_size = 0.25, random_state = RANDOM_STATE)


rmse_scoring = 'neg_root_mean_squared_error'
r2_scoring = 'r2'

# clone raw feature datasets
X_train_standarized = X_train.copy()
X_validation_standarized = X_validation.copy()

# Standardization - numerical features - ONLY.  NOT One-HOT Encoding Columns 
num_cols = ['precipitation', 'relative_irrigation', 'number_of_trips_reduction'] #['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']

# fit scaler on training data
standard_scaler = StandardScaler()
# apply standardization on numerical features ONLY
for i in num_cols:    
    # fit on training data column
    fit_standard_scaler = standard_scaler.fit(X_train_standarized[[i]])    
    # transform the training data column
    X_train_standarized[i] = fit_standard_scaler.transform(X_train_standarized[[i]])    
    # transform the testing data column
    X_validation_standarized[i] = fit_standard_scaler.transform(X_validation_standarized[[i]])

# configure the cross-validation procedure
cv_inner = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

outer_r2_results = list()
outer_rmse_results=list()

# define the model
# SVR Algorithm tuning
model = SVR()
cv_outer = RepeatedKFold(n_splits=15, n_repeats=15, random_state=RANDOM_STATE)

# Parameters for tuning
parameters = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'tol': [1e-6, 1e-5, 1e-4, 1e-3, 0.01],'C': [1, 1.5, 2, 2.5, 3]}]
#print("Tuning hyper-parameters")
# define search
search = GridSearchCV(model, parameters, cv = cv_inner, scoring = rmse_scoring, refit = True, n_jobs=-1)

time_start_grid_search = time.perf_counter()

# execute search
grid_result = search.fit(X_train_standarized, Y_train)
# get the best performing model fit on the whole training set
best_model = grid_result.best_estimator_
print('Best estimator')
print(best_model)
# grid_result.best_params_ is the same as search.best_params_
print("\nBest score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

cv = cross_val_score(search, X_train_standarized, Y_train, scoring=rmse_scoring, cv=cv_outer, n_jobs=-1)

time_end_grid_search = time.perf_counter()

msg = "Mean RMSE %f (%f)" % ( cv.mean(), cv.std())
print(msg)

time_start_predict_default = time.perf_counter()

# test predictions for the default hyperparams model on the holdout set   
model = SVR(kernel='rbf', tol=0.001, C=1)
fit = model.fit(X_train_standarized,Y_train)   
fit_score = fit.score(X_train_standarized,Y_train)  
prediction = model.predict(X_validation_standarized)    
r2_predict_score = metrics.r2_score(Y_validation,prediction)
RMSE_predict_score = metrics.mean_squared_error(Y_validation,prediction)

time_end_predict_default = time.perf_counter()

print('\n Default hyperparams')
print('R2 Training Score (Optimal is value 1)\n')  
print(round(fit_score, 5))

print('------------------------------------------------------')

print('R2 Test Score (Optimal is value 1)\n')  
print(round(r2_predict_score, 5))

print('------------------------------------------------------')

print('RMSE Score (Optimal is value 0)\n')    
print(round(RMSE_predict_score, 5))

time_start_predict_tuned = time.perf_counter()

# test predictions for the hyper tuned model on the holdout set   
model = SVR(kernel='rbf', tol=0.000001, C=1)
fit = model.fit(X_train_standarized,Y_train)
fit_score = fit.score(X_train_standarized,Y_train)     
prediction = model.predict(X_validation_standarized)    
r2_predict_score = metrics.r2_score(Y_validation,prediction)
RMSE_predict_score = metrics.mean_squared_error(Y_validation,prediction)

time_end_predict_tuned = time.perf_counter()

print('\n Default hyperparams')
print('R2 Training Score (Optimal is value 1)\n')  
print(round(fit_score, 5))

print('------------------------------------------------------')

print('R2 Test Score (Optimal is value 1)\n')  
print(round(r2_predict_score, 5))

print('------------------------------------------------------')

print('RMSE Score (Optimal is value 0)\n')    
print(round(RMSE_predict_score, 5))

print(f"Execution duration for SVMR hyperparameter tuning: {time_end_grid_search - time_start_grid_search:0.4f} seconds")
print(f"Execution duration for default SVMR fit & predict: {time_end_predict_default - time_start_predict_default:0.4f} seconds")
print(f"Execution duration for tuned SVMR fit & predict: {time_end_predict_tuned - time_start_predict_tuned:0.4f} seconds")