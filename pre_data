# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, runninga this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/Users/babakmodami/Desktop/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/babakmodami/Desktop/house-prices-advanced-regression-techniques/test.csv')

train.describe()

train.shape,test.shape

train.head(3)

train.shape,test.shape

#check for dupes for Id
idsUnique = len(set(train.Id))
idsTotal = train.shape[0]
idsdupe = idsTotal - idsUnique
print(idsdupe)
#drop id col
train.drop(['Id'],axis =1,inplace=True)

test.head(3)

#correlation matrix
corrmat = train.corr()
#f, ax = plt.subplots(figsize=(20, 9))
#sns.heatmap(corrmat, vmax=.8, annot=True);

# most correlated features
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
#plt.figure(figsize=(10,10))
#g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")


#sns.barplot(train.OverallQual,train.SalePrice)

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(train[cols], size = 2.5)
#plt.show();
