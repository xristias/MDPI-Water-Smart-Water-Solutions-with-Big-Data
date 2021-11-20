# Computing Step 3
# Feature engineering - Correlations

import numpy as np
from numpy.lib.function_base import average, diff
import pandas as pd
import random
from pandas import set_option

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


# correlation
set_option('precision', 23)
correlation_matrix = X.corr(method='pearson').abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]

print('Columns to drop')  
print(to_drop)

#ucorrelated_X = X.drop(to_drop, axis=1)
#print('Columns to keep')  
#print(ucorrelated_X.head)





