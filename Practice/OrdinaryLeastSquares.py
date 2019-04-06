X = [12,23,54,23,54,21,42]
Y = [32,23,77,64,12,35,23]

sumx, sumy, sum1, sum2 = 0, 0, 0, 0
length = len(X)

# Get the mean of x and y
for i in range(0, length):
    sumx += X[i]
    sumy += Y[i]
meanx = sumx / length
meany = sumy / length

for i in range(0, length):
    sum1 += (X[i] - meanx) * (Y[i] - meany)
    sum2 += (X[i] - meanx) ** 2

# m is slope of the line
m = sum1 / sum2
# b is the Y Intercept
b = meany - (m * meanx)

# Get Y values using y = mx + b formula
prediction = []
for i in range(0, length):
    prediction.append(m * X[i] + b)

print("\nmy results")
print("Slope :", m, "\nIntercept", b)
print("Predictions\n", prediction)

# Using LinearRegression from sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X1 = [[12],[23],[54],[23],[54],[21],[42]]
model.fit(X1, Y)

print("\nsklean results")
print("Slope :", model.coef_, "\nIntercept", model.intercept_)
print("Predictions\n", model.predict(X1))









