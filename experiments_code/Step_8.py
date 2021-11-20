# Computing Step 8
# Use uncorrelated data with proper transformation and the optimal repeated cv parameters
# after sensitivity analysis
# for ensemble methods
# Use the Standardization, the transformation method that worked well from previous steps

import numpy as np
import random
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost.sklearn import XGBRegressor
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


time_start = time.perf_counter()

num_of_dataset_training_columns = 6
array = dataset.values
X =  dataset.iloc[:,0:num_of_dataset_training_columns]
Y =  dataset.iloc[:,num_of_dataset_training_columns]
validation_size = 0.25
# Test options and evaluation metric
num_folds = 10
#seed = 7

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

# Spot-Check Algorithms
models = []
models.append(('XGB',XGBRegressor(random_state=RANDOM_STATE)))
models.append(('GBM',GradientBoostingRegressor(random_state=RANDOM_STATE)))
models.append(('RF',RandomForestRegressor(random_state=RANDOM_STATE)))
models.append(('ET',ExtraTreesRegressor(random_state=RANDOM_STATE)))

rkfold = RepeatedKFold(n_splits=15, n_repeats=15, random_state=RANDOM_STATE)
RMSE_predict_scores=[]
model_names = []

print('\n')
print('Repeated CV Sensitivity analysis - RMSE Scores (Optimal is value 0)\n') 
time_start = time.perf_counter()


# use the values of repeated cv that came first from previous sensitivity analysis
# evaluate each model in turn
results_idealCV = []
results_repeatedCV = []
results_mean_idealCV = []
results_mean_repeatedCV = []
model_names = []
for name, model in models:   
    cv_results = cross_val_score(model, X_train_standarized, Y_train, cv=rkfold, scoring=rmse_scoring, n_jobs=-1)
    results_repeatedCV.append(cv_results)    
    model_names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

time_end = time.perf_counter()
print(f"Execution duration: {time_end - time_start:0.4f} seconds")