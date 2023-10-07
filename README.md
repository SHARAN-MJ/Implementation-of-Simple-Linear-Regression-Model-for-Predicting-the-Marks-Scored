# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas,numpy,and sklearn.
2. Calculate the values for the training data set.
3. Calculate the values for the test data set.
4. Plot the graph for both the data sets and calculate for MAE,MSE and RMSE.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mukil kumar V
RegisterNumber:  212222230087

```
```
1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

2)
 df.tail()

3)
 x=df.iloc[:,:-1].values
x

4)
 y=df.iloc[:,1].values
y

5)
#splitting train and tst data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

6)
 y_test

7)
 #graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

8)
plt.scatter(x_test,y_test,color='aqua')
plt.plot(x_test,regressor.predict(x_test),color='black')
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

9)
 mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

## df.head()
![ML1](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/04b13a44-0138-4b7a-a0d8-fc7729167297)

## df.tail()
![ML2](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/d9863f02-c6da-4624-a95e-4bf7e38e2b22)

## ARRAY VALUES OF X
![ML3](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/e8453028-2ec8-460b-bb23-7cea4b693ce9)

## ARRAY VALUES FOR Y
![ML4](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/7e4185af-70ea-4ee1-a67b-0ac4160565b3)

## VALUES OF Y PREDICTION
![ML5](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/8915c8cd-cd8f-4b7c-8a37-a15d886905d7)

## ARRAY VALUES OF Y TEST
![ML6](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/8c75828d-cdce-4359-8a4a-0be8390ecd5c)

## TRAINING SET GRAPH
![ML7](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/15fcb028-b206-40b3-ad60-cf326af0ff3e)

## TEST SET GRAPH
![ML8](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/7ad629d3-cb46-44a1-8620-4cac9a171847)

## VALUES OF MSE,MAE,AND RMSE
![ML9](https://github.com/Roselinjovita/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104296/fab3399b-0a15-4735-9ea8-b50a13258aff)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
