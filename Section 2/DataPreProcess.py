import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# load Data from csv file
dataset = pd.read_csv('Data.csv')

# add independant data to x, all columns except purchased
x = dataset.iloc[:, :-1].values
# add dependant data to y, purchased column
y = dataset.iloc[:, 3].values
# Use SimpleImputer to put mean average values into null fields


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
'''
removes depreciated warning but creates object array
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
trans = make_column_transformer( (OneHotEncoder(),[0]),remainder="passthrough")
x = trans.fit_transform(x)
'''
# Turn country strings into numeric values


ct = ColumnTransformer([('Country', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

# Change purchased column into numeric values, 1 = Yes, 0 = No
labelencoderY = LabelEncoder()
y = labelencoderY.fit_transform(y)
# Split the data into Training and Test Sets

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)
