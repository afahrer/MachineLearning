# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from resources import PercentDiff

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# using iloc[:, 0] creates 1D array which does not work when fitting data
# using iloc[:, :-1] creates a 2d array
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Create Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# Get Coef and intercept values
model.fit(X_train, Y_train)

# Predict Value of Salary using y = coef_ * x + intercept_ formula
prediction = model.predict(X_test)
for i in range(len(prediction)):
    print("\n\nPrediction: $", '{0:.0f}'.format(prediction[i]),
          "\nActual Sal: $", '{0:.0f}'.format(Y_test[i]),
          "\nPercent Diff:", '{0:.2f}'.format(PercentDiff.get(prediction[i], Y_test[i])), "%")

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

