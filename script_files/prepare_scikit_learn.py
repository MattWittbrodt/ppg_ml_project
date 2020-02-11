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

# excel file containing data
filename=r'F:/Grants/Vaccarino/PPG/Project 1/MIPS_JAN.xlsx'

# read in excel file and process into scikit-learn dictionary compatable array
from pandas import ExcelWriter
from pandas import ExcelFile
import datetime
from dateutil.relativedelta import relativedelta

df = pd.read_excel(filename,sheet_name=1)
# create age array from DOB and enrollment date
#age = np.empty((rows,1))
age=[]
for x,y in zip(df['dob'],df['enrollmentdate']):
    xs = x.split('-')
    ys = y.split('-')
    datex = datetime.datetime(int(xs[0]),int(xs[1]),int(xs[2])) # dob
    datey = datetime.datetime(int(ys[0]),int(ys[1]),int(ys[2])) # enrollment date
    age.append(relativedelta(datey,datex).years)
df['age'] = age
# gather target names
#feature_names = list(df.columns)
#target_names = 'age'
##dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
feature_names = ['age', 'gender', 'hispanic', 'asian', 'nativeamerican',
                 'africanamerican', 'pacificislander', 'caucasian',
                 'schoolyears', 'hypertension', 'diabetes', 'smokehistory',
                 'bmi']
target_names = ['prefmd']
rows,cols = df.shape
data = np.empty((rows,len(feature_names)))# create numpy array add feature names
for x in range(0,len(feature_names)):
    data[:,x] = df.pop(feature_names[x])
target = df.pop(target_names[0])
# create the dataset dictionary
dataset = {}
dataset['data'] = data
dataset['target'] = target
dataset['target_names'] = target_names[0]
dataset['DESCR'] = 'n/a'
dataset['feature_names'] = feature_names
dataset['filename'] = filename
# some data preparation is needed.
# gender (1:female,2:male) -> 1:female, 0:male
# smokehistory (1:Never smoked, 2: Former smoker, 3: Current smoker) -> one-hot-encoding
# imputation of all continuous columns for missing data





