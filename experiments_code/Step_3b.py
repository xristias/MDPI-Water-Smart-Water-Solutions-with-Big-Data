# Computing Step 3b
# Compute scores using ucorrelated columns, from results of step 3a

import numpy as np
import random
import time
from numpy.lib.function_base import average, diff
import pandas as pd
from sklearn import metrics
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

num_of_dataset_training_columns = 6
array = dataset.values
X =  dataset.iloc[:,0:num_of_dataset_training_columns]
Y =  dataset.iloc[:,num_of_dataset_training_columns]
validation_size = 0.25

#stratified splitting
X_train, X_validation, Y_train, Y_validation = scsplit(X, Y, stratify = Y,
                                         test_size = 0.25, random_state = RANDOM_STATE)


# correlation
correlation_matrix = X.corr(method='pearson').abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
uncorrelated_X_train = X_train.drop(to_drop, axis=1) 
# In the case where dropping correlated values
# does not give better results (previous step) leave the dataset intact in future steps
# drop correlated features also in validation set
uncorrelated_X_validation = X_validation.drop(to_drop, axis=1) 
# In the case where dropping correlated values
# does not give better results (previous step) leave the dataset intact in future steps

mse_scoring = 'neg_root_mean_squared_error'
r2_scoring = 'r2_score'
# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('Bayes Ridge', BayesianRidge()))
models.append(('Ridge Regression', RidgeCV()))
models.append(('LASSO', Lasso(random_state=RANDOM_STATE)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=RANDOM_STATE)))
models.append(('SVMR', SVR()))

r2_fit_scores=[]
r2_predict_scores=[]
RMSE_predict_scores=[]

for name, model in models:
    fit = model.fit(uncorrelated_X_train,Y_train)
    r2_fit_scores.append(fit.score(uncorrelated_X_train, Y_train))
    prediction = model.predict(uncorrelated_X_validation)    
    # Calculate the R squared -- it is the accuracy for regression
    r2_predict_scores.append(metrics.r2_score(Y_validation,prediction))
    RMSE_predict_scores.append(metrics.mean_squared_error(Y_validation,prediction))

max_r2_predict_score=max(r2_predict_scores)
min_rmse_predict_score=min(RMSE_predict_scores)

print('Raw Data\n')  
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