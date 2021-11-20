# Computing Step 10
# Observations of Overall Improvement

import numpy as np
import random
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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

num_of_dataset_training_columns = 6
array = dataset.values
X =  dataset.iloc[:,0:num_of_dataset_training_columns]
Y =  dataset.iloc[:,num_of_dataset_training_columns]
validation_size = 0.25

#stratified splitting
X_train, X_validation, Y_train, Y_validation = scsplit(X, Y, stratify = Y,
                                         test_size = 0.25, random_state = RANDOM_STATE)

mse_scoring = 'neg_root_mean_squared_error'
r2_scoring = 'r2_score'
# Spot-Check Algorithms
models = []
models.append(('XGB',XGBRegressor(random_state=RANDOM_STATE)))
models.append(('GBM',GradientBoostingRegressor(random_state=RANDOM_STATE)))
models.append(('GBM Tuned',GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_STATE)))
models.append(('RF',RandomForestRegressor(random_state=RANDOM_STATE)))
models.append(('ET',ExtraTreesRegressor(random_state=RANDOM_STATE)))

#  Standarized data
r2_fit_scores=[]
r2_predict_scores=[]
RMSE_predict_scores=[]

# clone raw feature datasets
X_train_standarized = X_train.copy()
X_validation_standarized = X_validation.copy()

# numerical features - ONLY.  NOT One-HOT Encoding Columns 
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

for name, model in models:
    fit = model.fit(X_train_standarized,Y_train)
    # Calculate the R squared -- it is the accuracy for regression
    r2_fit_scores.append(fit.score(X_train_standarized, Y_train))
    prediction = model.predict(X_validation_standarized)    
    r2_predict_scores.append(metrics.r2_score(Y_validation,prediction))
    RMSE_predict_scores.append(metrics.mean_squared_error(Y_validation,prediction))

max_r2_predict_score=max(r2_predict_scores)
min_rmse_predict_score=min(RMSE_predict_scores)

print('\nStandarized Data\n')  
print('R2 Fit Scores (Optimal is value 1)\n')  
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

