import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset2019 = pd.read_csv('Fish.csv')

X = dataset2019.iloc[:, 4].values # Cross Lenght
y = dataset2019.iloc[:, 5].values # Height

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Cross Lenght(Training set)')
plt.xlabel('Cross Lenght')
plt.ylabel('Height')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Cross Lenght (Test set)')
plt.xlabel('Cross Lenght')
plt.ylabel('Height')
plt.show()