#!/usr/bin/env python
# coding: utf-8

#%% Loading packages
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))


#%% excel file containing data
filename = '/Users/mattwittbrodt/OneDrive - Emory University/machine_learning/data/MIPS_JAN.xlsx'
#filename = 'F:\Grants\Vaccarino\PPG\Project 1\MIPS_JAN.xlsx'

#%% read in excel file and process into scikit-learn dictionary compatable array
#from pandas import ExcelWriter
#from pandas import ExcelFile
import datetime
from dateutil.relativedelta import relativedelta

df = pd.read_excel(filename, sheet_name = 1)
rows,cols = df.shape

#%% Calculating Age
age=[]
for x,y in zip(df['dob'],df['enrollmentdate']):
    xs = x.split('-')
    ys = y.split('-')
    datex = datetime.datetime(int(xs[0]),int(xs[1]),int(xs[2])) # dob
    datey = datetime.datetime(int(ys[0]),int(ys[1]),int(ys[2])) # enrollment date
    age.append(relativedelta(datey,datex).years)
df['age'] = age


#%% gather target names
feature_names = ['age','gender', 'hispanic', 'asian', 'nativeamerican',
                 'africanamerican', 'pacificislander', 'caucasian',
                 'schoolyears', 'hypertension', 'diabetes', 'smokehistory',
                 'bmi', 'prefmd','bdi_sum','rest_systolic10minrest',
                 'rest_diastolic10minrest','rest_heartrate10minrest']
target_names = ['prefmd']


#%% New DataFrame with limited columns and fixing data types
data = df.loc[:,feature_names]
data['smokehistory'] = data['smokehistory'].astype('category')
data['hispanic'] = data['hispanic'].astype('category')
data['asian'] = data['asian'].astype('category')
data['nativeamerican'] = data['nativeamerican'].astype('category')
data['africanamerican'] = data['africanamerican'].astype('category')
data['caucasian'] = data['caucasian'].astype('category')
data['pacificislander'] = data['pacificislander'].astype('category')
data['hypertension'] = data['hypertension'].astype('category')
data['diabetes'] = data['diabetes'].astype('category')
data['gender'] = data['gender'].astype('category')

#%% Doing One Hot Encoding on Smokehistory
data_ohe = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_ohe.columns))

#%% Imputation - Is neded?

# Initial checks- NaN's and Zeros
data_ohe.isnull().values.any()
test = data_ohe.loc[:,['bmi','age','schoolyears','rest_heartrate10minrest',
                       'rest_diastolic10minrest','rest_systolic10minrest','bdi_sum']] == 0
test = test.loc[(test.bmi == True) | (test.age == True) | (test.schoolyears == True) | 
                 (test.rest_diastolic10minrest == True) | (test.rest_systolic10minrest == True) |
                 (test.rest_heartrate10minrest == True) | (test.bdi_sum == True),:]

# Setting Up Imputer
from sklearn.impute import SimpleImputer

# Since our missing values are all 0's, replacing with mean
imp = SimpleImputer(missing_values = 0, strategy="mean")
data_ohe[['bmi']] = imp.fit_transform(data_ohe[['bmi']]).ravel() # Need ravel for dimensions
data_ohe[['age']] = imp.fit_transform(data_ohe[['age']]).ravel() # Need ravel for dimensions
data_ohe[['schoolyears']] = imp.fit_transform(data_ohe[['schoolyears']]).ravel() # Need ravel for dimensions
data_ohe[['rest_diastolic10minrest']] = imp.fit_transform(data_ohe[['rest_diastolic10minrest']]).ravel() # Need ravel for dimensions
data_ohe[['rest_systolic10minrest']] = imp.fit_transform(data_ohe[['rest_systolic10minrest']]).ravel() # Need ravel for dimensions
data_ohe[['rest_heartrate10minrest']] = imp.fit_transform(data_ohe[['rest_heartrate10minrest']]).ravel() # Need ravel for dimensions
data_ohe[['bdi_sum']] = imp.fit_transform(data_ohe[['bdi_sum']]).ravel() # Need ravel for dimensions
data_ohe[['prefmd']] = imp.fit_transform(data_ohe[['prefmd']]).ravel()

# Checking again for 0's - should return a 0 x 3 Dataframe 
test2 = data_ohe.loc[:,['bmi','age','schoolyears','rest_heartrate10minrest',
                       'rest_diastolic10minrest','rest_systolic10minrest','bdi_sum']] == 0
test2 = test2.loc[(test2.bmi == True) | (test2.age == True) | (test2.schoolyears == True) |
                  (test2.rest_diastolic10minrest == True) | (test2.rest_systolic10minrest == True) |
                  (test2.rest_heartrate10minrest == True) | (test2.bdi_sum == True),:]

#%% Fixing Gender ((1:female,2:male) -> 1:female, 0:male) INDEXING AND WRITING OVER IN PANDAS df PROMPTS A WARNING
#gendr = data_ohe.pop('gender')
#for ii in range(0,len(gendr)):
#    if gendr[ii] == 2:
#        gendr[ii] = 0
        
# insert new gender array to data_ohe
#data_ohe['gender'] = gendr

## create a scatter matrix from the dataframe
#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
##scat = scatter_matrix(data_ohe, alpha=0.5, figsize=(10, 10), diagonal='kde')
##plt.show() # not very helpful
#
## splitting the dataframe into two, still not very telling with many binary variables
#
#dfs = np.split(data_ohe, [8], axis=1)
#scat1 = scatter_matrix(dfs[0], alpha=0.5, figsize=(10, 10), diagonal='kde')
##plt.show(block = False)
#scat2 = scatter_matrix(dfs[1], alpha=0.5, figsize=(10, 10), diagonal='kde')
##plt.show()

#%% create the dataset dictionary

dataset = {}
dataset['target'] = data_ohe.pop(target_names[0])
dataset['data'] = data_ohe.values
dataset['target_names'] = target_names[0]
dataset['DESCR'] = 'n/a'
dataset['feature_names'] = data_ohe.columns
dataset['filename'] = filename

# split data for scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

#%% KNN Regression
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=10)
reg.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(reg.score(X_test, y_test)))
knn_predictions = reg.predict(X_test)

#%% Linear Regression    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))

lr_predictions = lr.predict(X_test)

#%% Mutiplotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2, figsize=(10,3))

plt.subplot(1,2,1)
plt.scatter(y_test, knn_predictions, alpha = 0.7)
plt.title('KNN (n = 3) Predictions vs Actual')
plt.xlabel('Actual Test FMD')
plt.ylabel('Predicted Test FMD')

plt.subplot(1,2,2)
plt.scatter(y_test, lr_predictions, alpha = 0.7)
plt.title('Linear Regression Predictions vs Actual')
plt.xlabel('Actual Test FMD')
plt.ylabel('Predicted Test FMD')

plt.show()

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(tree.score(X_test, y_test)))

tree_pred = tree.predict(X_test)
#plt.scatter(y_test, tree_pred, alpha = 0.7)
#plt.title('Decision Tree Regression Predictions vs Actual')
#plt.xlabel('Actual Test FMD')
#plt.ylabel('Predicted Test FMD')

def plot_feature_importances_cancer(model):
    n_features = dataset['data'].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), dataset['feature_names']) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(tree)

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=100)
forest.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(forest.score(X_test, y_test)))

def plot_feature_importances_cancer(model):
    n_features = dataset['data'].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), dataset['feature_names']) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(forest)






