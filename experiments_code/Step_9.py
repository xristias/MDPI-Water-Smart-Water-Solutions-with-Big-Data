# Computing Step 9
# Hyperparameter Tuning (Gradient Boosting algorithm) 
# Use uncorrelated data with proper transformation (Standardization) and the optimal repeated cv parameters
# after sensitivity analysis

import numpy as np
import random
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
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

#stratified splitting
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
cv_outer = RepeatedKFold(n_splits=15, n_repeats=15, random_state=RANDOM_STATE)

outer_r2_results = list()
outer_rmse_results=list()

# define the model
# Gradient Boosting Algorithm tuning
model = GradientBoostingRegressor(random_state=RANDOM_STATE)
print('Default estimator')
print(model.get_params())

# Parameters for tuning
loss = ['quantile', 'huber']
num_estimators = [50,200]
learn_rates = [0.1, 0.9]
max_depths = [2, 4]
min_samples_leaf = [1,3]
min_samples_split = [2,4]

param_grid = {
              #'loss' : loss,
              'n_estimators': num_estimators,
              #'learning_rate': learn_rates,
              #'max_depth': max_depths,
              #'min_samples_leaf': min_samples_leaf,
              #'min_samples_split': min_samples_split
              }

#print("Tuning hyper-parameters")
time_start = time.perf_counter()

# define search
search = GridSearchCV(model, param_grid, cv = cv_inner, scoring = rmse_scoring, refit = True, n_jobs=-1)
# execute search
grid_result = search.fit(X_train_standarized, Y_train)
# get the best performing model fit on the whole training set
best_model = grid_result.best_estimator_
print('Best estimator')
print(best_model)
# grid_result.best_params_ is the same as search.best_params_
print("\nBest score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

cv = cross_val_score(search, X_train_standarized, Y_train, scoring=rmse_scoring, cv=cv_outer, n_jobs=-1)
msg = "Mean RMSE %f (%f)" % ( cv.mean(), cv.std())
print(msg)

time_end = time.perf_counter()

print(f"Execution duration: {time_end - time_start:0.4f} seconds")