# Computing Step 5
# Normalization or Standarization with default cross validation

# Standardization is used, the transformation method that worked well from step 4

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
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

# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('Bayes Ridge', BayesianRidge()))
models.append(('Ridge Regression', RidgeCV()))
models.append(('LASSO', Lasso(random_state=RANDOM_STATE)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=RANDOM_STATE)))
models.append(('SVMR', SVR()))

# chose from previous step the appropriate tranformer
pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('Bayes\nRidge', Pipeline([('Scaler', StandardScaler()),('Bayes Ridge',
BayesianRidge())])))
pipelines.append(('Ridge\nRegression', Pipeline([('Scaler', StandardScaler()),('Ridge Regression',
RidgeCV())])))
pipelines.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('SVMR', Pipeline([('Scaler', StandardScaler()),('SVMR',
SVR())])))


# step 5 mean accuracy results standard cross validation
skfold = KFold(n_splits=num_folds, random_state=RANDOM_STATE)    
RMSE_predict_scores=[]
model_names = []

print('\n')
print('Standard CV - RMSE Scores (Optimal is value 0)\n') 
cv_results = []
model_names = []
for name, model in models:    
    cv_results = cross_val_score(model, X_train_standarized, Y_train, cv=skfold, scoring=rmse_scoring, n_jobs=-1)
    RMSE_predict_scores.append(cv_results)
    model_names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms -- Produces Figure 1 (Experiment 5)
font = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 16}

pyplot.rc('font', **font)
fig = pyplot.figure()
#fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(RMSE_predict_scores)
ax.set_xticklabels(model_names)
pyplot.xticks(rotation=45)
pyplot.subplots_adjust(bottom=0.2)
pyplot.show()