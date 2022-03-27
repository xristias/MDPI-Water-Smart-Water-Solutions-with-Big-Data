# Computing Step 6
# Sensitivity Analysis on cross validation

from shutil import which
import numpy as np
import random
import time
import pandas as pd
from scipy.stats import sem
from numpy import mean
from scipy.stats import pearsonr
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
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
from numpy import polyfit
from numpy import asarray

# Set a Random State value
RANDOM_STATE = 7
# Set Python random a fixed value
random.seed(RANDOM_STATE)
# Set numpy random a fixed value
np.random.seed(RANDOM_STATE)


# Load dataset
filename = 'experiments_code/executionset_regression_1.csv'
names = ['management', 'soil_type.CL', 'soil_type.LS', 'soil_type.SL', 'precipitation',
         'relative_irrigation', 'number_of_trips_reduction', 'relative_profit_percentage']
dataset = pd.read_csv(filename, delim_whitespace=True, header=0)

font = {'family': 'arial',
        'weight': 'bold',
        'size': 16}

pyplot.rc('font', **font)

time_start_sensitivity_analysis_cv = time.perf_counter()

num_of_dataset_training_columns = 6
array = dataset.values
X = dataset.iloc[:, 0:num_of_dataset_training_columns]
Y = dataset.iloc[:, num_of_dataset_training_columns]
validation_size = 0.25

# stratified splitting
X_train, X_validation, Y_train, Y_validation = scsplit(X, Y, stratify=Y,
                                                       test_size=0.25, random_state=RANDOM_STATE)

# Test options and evaluation metric
num_folds = 10
rmse_scoring = 'neg_root_mean_squared_error'
r2_scoring = 'r2'

# clone raw feature datasets
X_train_standarized = X_train.copy()
X_validation_standarized = X_validation.copy()

# Standardization - numerical features - ONLY.  NOT One-HOT Encoding Columns
num_cols = ['precipitation', 'relative_irrigation',
            'number_of_trips_reduction']

# fit scaler on training data
standard_scaler = StandardScaler()
# apply standardization on numerical features ONLY
for i in num_cols:
    # fit on training data column
    fit_standard_scaler = standard_scaler.fit(X_train_standarized[[i]])
    # transform the training data column
    X_train_standarized[i] = fit_standard_scaler.transform(
        X_train_standarized[[i]])
    # transform the testing data column
    X_validation_standarized[i] = fit_standard_scaler.transform(
        X_validation_standarized[[i]])


# sensitivity analysis for k (regular)
# evaluate a model with a given number of repeats
def evaluate_model_regularCV(X_train_standarized, Y_train, folds):
    # prepare the cross-validation procedure
    cv = KFold(n_splits=folds, random_state=RANDOM_STATE)
    # create Linear Regression model
    model = LinearRegression()

    # evaluate model
    scores = cross_val_score(model, X_train_standarized,
                             Y_train, cv=cv, scoring=rmse_scoring, n_jobs=-1)
    return scores


model = LinearRegression()
ideal = cross_val_score(model, X_train_standarized, Y_train,
                        cv=LeaveOneOut(), scoring=rmse_scoring, n_jobs=-1)

meanIdeal = mean(ideal)
print('Ideal: %.7f' % meanIdeal)
sensitivity_folds = range(2, 41)
# record mean and min/max of each set of results
means, mins, maxs, results_regularCV, ideal_distances, std_errors = list(
), list(), list(), list(), list(), list()
# evaluate each k value
for k in sensitivity_folds:
    # define the test condition
    cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    # evaluate k value
    sensitivity_scores = evaluate_model_regularCV(
        X_train_standarized, Y_train, k)
    # store scores
    results_regularCV.append(sensitivity_scores)
    # store std error
    std_error = sem(sensitivity_scores)
    std_errors.append(std_error)
    # store mean accuracy
    meanValue = mean(sensitivity_scores)
    means.append(mean)
    # store min and max relative to the mean
    mins.append(meanValue - sensitivity_scores.min())
    maxs.append(sensitivity_scores.max() - meanValue)
    # store difference from ideal
    difference = abs(meanValue - meanIdeal)
    ideal_distances.append(difference)

    # report performance
    print('> folds=%d, ideal_diff=%.8f, accuracy=%.8f (%.8f,%.8f), stderr=%.8f ' % (
        k, difference, meanValue, sensitivity_scores.min(), sensitivity_scores.max(), std_error))

min_ideal_difference = min(ideal_distances)
min_ideal_difference_index = ideal_distances.index(min_ideal_difference)

# find the best (lowest) std errror
min_std_error = min(std_errors)
# find the max std error
max_std_error = max(std_errors)
# find the index in the list of the minimum value above
min_std_error_index = std_errors.index(min_std_error)
# find the index in the list of the maximum value above
max_std_error_index = std_errors.index(max_std_error)

time_end_sensitivity_analysis_cv = time.perf_counter()

print('Best related to ideal: %.8f at position: %d for fold: %d' % (min_ideal_difference,
                                                                    min_ideal_difference_index, sensitivity_folds[min_ideal_difference_index]))
print('Std Error value for the best accuracy score: %.8f' %
      (std_errors[min_ideal_difference_index]))

print('Best std error value %.8f at position: %d for fold: %d' % (
    min_std_error, min_std_error_index, sensitivity_folds[min_std_error_index]))
print('Worst std error value %.8f at position: %d for fold: %d' % (
    max_std_error, max_std_error_index, sensitivity_folds[max_std_error_index]))
print('Std error mean value %.8f' % (mean(std_errors)))

# CV Sensitivity Analysis -- Produces Figure 2 (Experiment 6a)
# plot the ideal case in a separate color
pyplot.axhline(y=meanIdeal, color='r', linestyle='-')

flierprops = dict(marker='o', markerfacecolor='green',
                  markersize=12, markeredgecolor='none')
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='solid', linewidth=2.5, color='blue')
pyplot.boxplot(results_regularCV, labels=[str(
    r) for r in sensitivity_folds], showmeans=True, meanprops=meanlineprops, meanline=True)
# label the plot
pyplot.xlabel('Number of Folds (k)', fontsize=16,
              fontweight='bold', labelpad=30)
pyplot.ylabel('Mean Accuracy (CV)', fontsize=16,
              fontweight='bold', labelpad=30)
# show the plot
pyplot.show()


# sensitivity analysis for k (repeated cv)
# evaluate a model with a given number of repeats
def evaluate_model_repeatedCV(X_train_standarized, Y_train, repeats, folds):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats,
                       random_state=RANDOM_STATE)
    # create Linear Regression model
    model = LinearRegression()
    # evaluate model
    scores = cross_val_score(model, X_train_standarized,
                             Y_train, cv=cv, scoring=rmse_scoring, n_jobs=-1)
    return scores


time_start_sensitivity_analysis_rcv = time.perf_counter()

# record mean and min/max of each set of results
means2, mins2, maxs2, results_repeatedCV, ideal_distances2, std_errors2, folds_repeats_verbal_legend = list(
), list(), list(), list(), list(), list(), list()
# test folds and repeates best value
min_fold = 10
max_fold = 15
min_repeat = 10
max_repeat = 20
folds = range(min_fold, max_fold + 1)
repeats = range(min_repeat, max_repeat + 1)
for f in folds:
    for r in repeats:
        # evaluate using a given number of repeats
        scores = evaluate_model_repeatedCV(X_train_standarized, Y_train, r, f)

        # store scores
        results_repeatedCV.append(scores)
        # store std error
        std_error2 = sem(scores)
        std_errors2.append(std_error2)
        # store mean accuracy
        meanValue2 = mean(scores)
        means2.append(meanValue2)
        # store difference from ideal
        difference2 = abs(meanValue2 - meanIdeal)
        ideal_distances2.append(difference2)

        # create a list mainly to populate x axis for folds & repeats
        # it contains the specific fold and repeat for the nested loop
        folds_repeats_verbal_legend.append("Fol:%d Rep:%d" % (f, r))

        # report performance
        print('> fold-repeat=%s, ideal_diff=%.8f, accuracy=%.8f (%.8f,%.8f), stderr=%.8f ' %
              (folds_repeats_verbal_legend[-1], difference2, meanValue2, scores.min(), scores.max(), std_error2))

# find the best (lowest) accuracy
min_ideal_difference2 = min(ideal_distances2)
# find the index in the list of the minimum value above
min_ideal_difference_index2 = ideal_distances2.index(min_ideal_difference2)


# find the best (lowest) std errror
min_std_error2 = min(std_errors2)
# find the max std error
max_std_error2 = max(std_errors2)
# find the index in the list of the minimum value above
min_std_error_index2 = std_errors2.index(min_std_error2)
# find the index in the list of the maximum value above
max_std_error_index2 = std_errors2.index(max_std_error2)

time_end_sensitivity_analysis_rcv = time.perf_counter()

print('Best accuracy difference related to ideal: %.8f at position (fold & repeat): %s' %
      (min_ideal_difference2, folds_repeats_verbal_legend[min_ideal_difference_index2]))
print('Std Error value for the best accuracy score: %.8f' %
      (std_errors2[min_ideal_difference_index2]))

print('Best std error value %.8f at position (fold & repeat): %s' %
      (min_std_error2, folds_repeats_verbal_legend[min_std_error_index2]))
print('Worst std error value %.8f at position (fold & repeat): %s' %
      (max_std_error2, folds_repeats_verbal_legend[max_std_error_index2]))
print('Std error mean value %.8f' % (mean(std_errors2)))

# Repeated CV Sensitivity Analysis -- Produces Figure 3 (Experiment 6b)
# plot the ideal case in a separate color
pyplot.axhline(y=meanIdeal, color='r', linestyle='-')

flierprops = dict(marker='o', markerfacecolor='green',
                  markersize=12, markeredgecolor='none')
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='solid', linewidth=2.5, color='blue')
pyplot.boxplot(results_repeatedCV, labels=[str(
    r) for r in folds_repeats_verbal_legend], showmeans=True, meanprops=meanlineprops, meanline=True)

pyplot.xticks(rotation=270)
pyplot.subplots_adjust(bottom=0.2)
pyplot.xlabel('Number of Folds (k) and Repeats (n)', fontsize=16,
              fontweight='bold', labelpad=30)
pyplot.ylabel('Mean Accuracy (RCV)', fontsize=16,
              fontweight='bold', labelpad=50)
pyplot.show()

# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('Bayes Ridge', BayesianRidge()))
models.append(('Ridge Regression', RidgeCV()))
models.append(('LASSO', Lasso(random_state=RANDOM_STATE)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=RANDOM_STATE)))
models.append(('SVMR', SVR()))

# mean accuracy results for standard cross validation, no sensitivity analysis
skfold = KFold(n_splits=num_folds, random_state=RANDOM_STATE)
RMSE_predict_scores = []
model_names = []

print('\n')
print('Standard CV - RMSE Scores (Optimal is value 0)\n')
time_start_cv = time.perf_counter()

cv_results = []
model_names = []
for name, model in models:
    cv_results = cross_val_score(
        model, X_train_standarized, Y_train, cv=skfold, scoring=rmse_scoring, n_jobs=-1)
    RMSE_predict_scores.append(cv_results)
    model_names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

time_end_cv = time.perf_counter()

# mean accuracy results after sensitivity analysis
rkfold = RepeatedKFold(n_splits=15, n_repeats=15, random_state=RANDOM_STATE)
RMSE_predict_scores = []
model_names = []

print('\n')
print('Repeated CV Sensitivity analysis - RMSE Scores (Optimal is value 0)\n')

time_start_rcv = time.perf_counter()

# use the values of repeated cv that came first from previous sensitivity analysis
# evaluate each model in turn
results_idealCV = []
results_repeatedCV = []
results_mean_idealCV = []
results_mean_repeatedCV = []
model_names = []
for name, model in models:
    cv_ideal_results = cross_val_score(
        model, X_train_standarized, Y_train, cv=rkfold, scoring=rmse_scoring, n_jobs=-1)
    results_idealCV.append(cv_ideal_results)
    results_mean_idealCV.append(mean(cv_ideal_results))

    cv_results = cross_val_score(
        model, X_train_standarized, Y_train, cv=rkfold, scoring=rmse_scoring, n_jobs=-1)
    results_repeatedCV.append(cv_results)
    results_mean_repeatedCV.append(mean(cv_results))
    model_names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

time_end_rcv = time.perf_counter()

corr, _ = pearsonr(results_mean_repeatedCV, results_mean_idealCV)
print('Correlation: %.3f' % corr)

# Correlation for all algorithms -- Produces Figure 4 (Experiment 6c)
# scatter plot of results
pyplot.scatter(results_mean_repeatedCV, results_mean_idealCV)
# plot the line of best fit
coeff, bias = polyfit(results_mean_repeatedCV, results_mean_idealCV, 1)
line = coeff * asarray(results_mean_repeatedCV) + bias
pyplot.plot(results_mean_repeatedCV, line, color='r')
# label the plot
pyplot.xlabel('Mean Accuracy (Repeated CV)', fontsize=16,
              fontweight='bold', labelpad=15)
pyplot.ylabel('Mean Accuracy (LOOCV)', fontsize=16,
              fontweight='bold', labelpad=15)
# show the plot
pyplot.show()


# Compare Algorithms -- Produces Figure 5 (Experiment 6d)
fig = pyplot.figure()
ax = fig.add_subplot(111)
pyplot.boxplot(results_repeatedCV)
ax.set_xticklabels(model_names)
pyplot.xticks(rotation=45)
pyplot.subplots_adjust(bottom=0.2)
# label the plot
pyplot.ylabel('Mean Accuracy (RCV)', fontsize=16,
              fontweight='bold', labelpad=30)
pyplot.show()


print(
    f"Execution duration for standard cross validation sensitivity analysis: {time_end_sensitivity_analysis_cv - time_start_sensitivity_analysis_cv:0.4f} seconds")
print(
    f"Execution duration for repeated cross validation sensitivity analysis: {time_end_sensitivity_analysis_rcv - time_start_sensitivity_analysis_rcv:0.4f} seconds")
print(
    f"Execution duration for standard cross validation on algorithms: {time_end_cv - time_start_cv:0.4f} seconds")
print(
    f"Execution duration repeated cross validation on algorithms: {time_end_rcv - time_start_rcv:0.4f} seconds")
