#sachinyadav3496 git
# Languages_can_be_used ==> python, r, scala. matlab, octave, sas, juila

# Python_required ==> datatype  *  operator  *  conditional statement  *  loop  *  function

import pandas as pd

#reading csv
data = pd.read_csv("A:\\Workshop\\Python\\h1b.csv")
data.head()

#decribe to states of data info.
data.describe()
data.columns
data.shape

#top 20 companies having h1b applicants
data['EMPLOYER_NAME'].value_counts()[:20]

#top companies which provide max. privallages wages
data.columns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

new_data = data.groupby('EMPLOYER_NAME').agg({
    "PREVAILING_WAGE":np.mean
})

new_data.head()

new_data.sort_values(by='PREVAILING_WAGE')
new_data.sort_values(by='PREVAILING_WAGE', ascending=False)[:20].plot(kind='bar')
plt.show()

#split set into train test
from sklearn.model_selection import train_test_split

#model whic learn from data
from sklearn.linear_model import LinearRegression

from sklearn import datasets

house = datasets.load_boston()

#housing price prediction

#step1  ==>  prepare your data
house.feature_names
print(house.DESCR)

features = pd.DataFrame(house.data, columns=house.feature_names)
features.head()

target = pd.Series(house.target)
target.head()

import seaborn as sns
from sklearn.metrics.regression import r2_score

sns.scatterplot(x=features['CRIM'],y=target)
plt.show()

for column_name in features.columns:
    sns.scatterplot(x=features[column_name],y=target)
    plt.show()

# splitting into training and test data
train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.3)

#select representation or model
model = LinearRegression()

# let's train our model
model.fit(train_X,train_Y)

m = model.coef_
m
c = model.intercept_
c

features.loc[3,]

#predict price
model.predict(features.loc[3:5,])

#actual price
target.loc[3:5]

pred_y = model.predict(test_X)

r2_score(test_Y, pred_y)

#step2  ==>  

