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
                 'statinslipid', 'bmi','bdi_sum','rest_systolic10minrest',
                 'rest_diastolic10minrest','rest_heartrate10minrest',
                 'restscan_diastolicvolume','restscan_systolicvolume',
                 'restscan_ejectfraction','restscan_wallthickenabnormality','il6_baseline', 'mcp1_baseline', 
                 'mmp_9_baseline', 'sdf_baseline', 'vegf_baseline', 'crp_baseline',
                 'icam1_baseline', 'vcam1_baseline', 'dyslipidemia','cad_mi',
                 'heartfailure','scid_dep','creatinine','troponin_base',
                 'antidepressant', 'nitrates', 'nsaids','warfarin',
                 'antiarrhythmic','antipsychotics','aceinhibitor','angiotensinreceptor',
                 'diuretics'	,'vasodilator' ,'anxiolytics' ,'aspirin' ,'betablocker',
                 'calciumantagonist' ,'clopidogrel' ,'diabetesmedication',
                 'digitalis','hormonereplacement',
                 'spectrest_basantscorerst','spectrest_basantsepscorerst','spectrest_basinfsepscorerst',
                 'spectrest_basinflatscorerst', 'spectrest_basantlatscorerst','spectrest_midantscorerst',
                 'spectrest_midantsepscorerst','spectrest_midinfsepscorerst',
                 'spectrest_apantscorerst',
                 'spectrest_apsepscorerst','spectrest_apinfscorerst','spectrest_aplatscorerst',
                 'spectrest_apscorerst', 'spectrest_rsscore','spectrest_mass','spectrest_ef',
                 'spectrest_edv','spectrest_esv','spectrest_sv','spectrest_phase_peak','spectrest_phase_sd',
                 'spectrest_phase_bandwidth','rest_wbc','rest_ly','rest_rbc','rest_hgb']
#spectrest_midinfscorerst,spctrest_midantlatscorerst,spctrest_midinflatscorerst

target_names = ['troponin_base']


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
data['statinslipid'] = data['statinslipid'].astype('category')
data['cad_mi'] = data['cad_mi'].astype('category')
data['dyslipidemia'] = data['dyslipidemia'].astype('category')
data['heartfailure'] = data['heartfailure'].astype('category')
data['scid_dep'] = data['scid_dep'].astype('category')
data['restscan_wallthickenabnormality'] = data['restscan_wallthickenabnormality'].astype('category')
data['antidepressant'] = data['antidepressant'].astype('category')
data['nitrates'] = data['nitrates'].astype('category')
data['nsaids'] = data['nsaids'].astype('category')
data['warfarin'] = data['warfarin'].astype('category')
data['antiarrhythmic'] = data['antiarrhythmic'].astype('category')
data['antipsychotics'] = data['antipsychotics'].astype('category')
data['aceinhibitor'] = data['aceinhibitor'].astype('category')
data['angiotensinreceptor'] = data['angiotensinreceptor'].astype('category')
data['diuretics'] = data['diuretics'].astype('category')
data['vasodilator'] = data['vasodilator'].astype('category')
data['anxiolytics'] = data['anxiolytics'].astype('category')
data['betablocker'] = data['betablocker'].astype('category')
data['calciumantagonist'] = data['calciumantagonist'].astype('category')
data['clopidogrel'] = data['clopidogrel'].astype('category')
data['diabetesmedication'] = data['diabetesmedication'].astype('category')
data['digitalis'] = data['digitalis'].astype('category')
data['hormonereplacement'] = data['hormonereplacement'].astype('category')



#%% Doing One Hot Encoding on Smokehistory
data_ohe = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_ohe.columns))

#%% Imputation - Is neded?
# Continuous Variables
continuous_var = ['age', 'schoolyears', 'bmi','bdi_sum','rest_systolic10minrest',
                 'rest_diastolic10minrest','rest_heartrate10minrest',
                 'restscan_diastolicvolume','restscan_systolicvolume',
                 'restscan_ejectfraction', 'troponin_base',
                  'il6_baseline', 'mcp1_baseline', 
                 'mmp_9_baseline', 'sdf_baseline', 'vegf_baseline', 'crp_baseline',
                 'icam1_baseline', 'vcam1_baseline', 'creatinine',
                 'troponin_base',
                 'rest_wbc','rest_ly','rest_rbc','rest_hgb']

# Initial checks- NaN's and Zeros
test = data_ohe.loc[:,continuous_var] == 0
test = test[test[:] == True]

# Setting Up Imputer
from sklearn.impute import SimpleImputer

# Since our missing values are all 0's, replacing with mean
imp = SimpleImputer(missing_values = 0, strategy="mean")

for var in range(len(continuous_var)):
    n = continuous_var[var]
    data_ohe[[n]] = imp.fit_transform(data_ohe[[n]]).ravel() # Need ravel for dimensions

# Checking again for 0's - should return a 0 x 3 Dataframe 
test2 = data_ohe.loc[:,continuous_var] == 0
test2 = test2[test2[:] == True]

#%%

spect_var = ['spectrest_basantscorerst','spectrest_basantsepscorerst','spectrest_basinfsepscorerst',
                 'spectrest_basinflatscorerst', 'spectrest_basantlatscorerst','spectrest_midantscorerst',
                 'spectrest_midantsepscorerst','spectrest_midinfsepscorerst','spectrest_midinfscorerst',
                 'spctrest_midinflatscorerst','spctrest_midantlatscorerst','spectrest_apantscorerst',
                 'spectrest_apsepscorerst','spectrest_apinfscorerst','spectrest_aplatscorerst',
                 'spectrest_apscorerst', 'spectrest_rsscore','spectrest_mass','spectrest_ef',
                 'spectrest_edv','spectrest_esv','spectrest_sv','spectrest_phase_peak','spectrest_phase_sd',
                 'spectrest_phase_bandwidth']
spect_test = data_ohe.loc[:,spect_var] == np.nan
spect_test = spect_test[spect_test[:] == True]

# Since our missing values are all 0's, replacing with mean
imp = SimpleImputer(missing_values = np.nan, strategy="mean")

for var in spect_var:
    n = spect_var[var]
    data_ohe[[n]] = imp.fit_transform(data_ohe[[n]]).ravel() # Need ravel for dimensions


#%% Log Transfrom Troponin & other blood variables
data_ohe['mmp_9_baseline'] = np.log(data_ohe['mmp_9_baseline'])
data_ohe['il6_baseline'] = np.log(data_ohe['il6_baseline'])
data_ohe['mcp1_baseline'] = np.log(data_ohe['mcp1_baseline'])
data_ohe['sdf_baseline'] = np.log(data_ohe['sdf_baseline'])
data_ohe['vegf_baseline'] = np.log(data_ohe['vegf_baseline'])
data_ohe['crp_baseline'] = np.log(data_ohe['crp_baseline'])
data_ohe['icam1_baseline'] = np.log(data_ohe['icam1_baseline'])
data_ohe['vcam1_baseline'] = np.log(data_ohe['vcam1_baseline'])
data_ohe['troponin_base'] = np.log(data_ohe['troponin_base'])

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
print("Training set accuracy: {:.2f}".format(reg.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(reg.score(X_test, y_test)))
knn_predictions = reg.predict(X_test)

#%% Linear Regression    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print("Training set accuracy: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))

lr_predictions = lr.predict(X_test)

#%% Mutiplotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2, figsize=(10,3))

plt.subplot(1,2,1)
plt.scatter(y_test, knn_predictions, alpha = 0.7)
plt.title('KNN (n = 3) Predictions vs Actual')
plt.xlabel('Actual Test Troponin')
plt.ylabel('Predicted Test Troponin')

plt.subplot(1,2,2)
plt.scatter(y_test, lr_predictions, alpha = 0.7)
plt.title('Linear Regression Predictions vs Actual')
plt.xlabel('Actual Test Troponin')
plt.ylabel('Predicted Test Troponin')

plt.show()

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
print("Training set accuracy: {:.2f}".format(tree.score(X_train, y_train)))
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

forest = RandomForestRegressor(n_estimators=500, max_depth = 10, min_samples_split = 10)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(forest.score(X_test, y_test)))
#plot_feature_importances_cancer(forest)

random_pred = forest.predict(X_test)

plt.scatter(y_test, random_pred, alpha = 0.7)
plt.title('Random Forest Regression Predictions vs Actual')
plt.xlabel('Actual Test Troponin')
plt.ylabel('Predicted Test Troponin')

#%% Gradient Boosted Regression
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(random_state=0, n_estimators=1000, learning_rate=0.01, max_depth=3)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.sfcore(X_test, y_test)))

plot_feature_importances_cancer(gbrt)

#%% GraphViz
import graphviz
from sklearn.tree import export_graphviz

export_graphviz(gbrt.estimators_[99,0], out_file = "tree.dot", feature_names=dataset['feature_names'], 
               impurity=False, filled=True)

with open("tree.dot") as f: 
    dot_graph = f.read()

graphviz.Source(dot_graph)

#%% SVM with kernel
from sklearn.svm import SVR
svm = SVR(kernel='rbf', C=10, gamma=1).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))







