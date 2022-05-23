#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#reading csv file
df=pd.read_csv("Canada per capita.csv")

#cleaning the data
df=df.dropna()
print(df)
#changing the row name
df = df.rename(columns={"per capita income ":"capita"})

#plotting the model
plt.xlabel("Year")
plt.ylabel("Per capita")
plt.scatter(df.Year,df.capita,color='green',marker="+")
plt.show()

#checking the correlation
p=df.corr()
print(p)

#creating an linear regression object
reg=linear_model.LinearRegression()

#training the model
reg.fit(df[['Year']],df.capita)
x=reg.predict([[2020]])#prediction
print(x)