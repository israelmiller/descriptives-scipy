#import scipy, pandas, and numpy scipy
import pandas as pd
import numpy as np
import seaborn
from scipy import stats
from pandas import plotting
from statsmodels.formula.api import ols
#reading data from a csv file
data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")

#Creating a dataframe from numpy arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

#Manipulating data

data.shape #dimensions of the dataframe

data.columns #column names

print(data['Gender'])#print the data in a specific column

data[data['Gender'] == 'Female']['VIQ'].mean() #simpler selector

#groupby splits a dataframe on categorical variables.

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender,value.mean()))

groupby_gender.mean()

#plotting data

plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])

plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

#Hypothesis testing

#1 sample ttest
stats.ttest_1samp(data['VIQ'], 0)

#2 sample ttest
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender']== 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

#Paired tests: repeated measurements on the same individuals

#test FSIQ and PIQ to see if they are significantly different

stats.ttest_ind(data['FSIQ'], data['PIQ'])

#The above approach does not account for the fact that the same individuals are measured twice.
#this variance is accounted for in the paired ttest, or repeated measures ttest.

stats.ttest_rel(data['FSIQ'], data['PIQ'])

#This is equivalent to a 1-sample ttest on the difference between the two variables.

stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)

#t-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test to relax this assumption.

stats.wilcoxon(data['FSIQ'], data['PIQ'])

#Linear models, multiple factors, and analysis of variance

#simple linear regression

x = np.linspace(-5, 5, 20)

np.random.seed(1)

#normal distributed noise
y = -5 + 3 * x + 4 * np.random.normal(size=x.shape)

#create a dataframe with all the relevant variables
data = pd.DataFrame({'x': x, 'y': y})

#Specify an OLS and fir it:

model = ols("y ~ x", data).fit()

print(model.summary())

#Categorical variables: comparing groups or multiple categories

data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")

#we can write a comparison between IQ of male and female useing a linear model:

model = ols("VIQ ~ Gender + 1", data).fit()
#columns can be forced to be treated as categorical by putting C("") around the variable
# "+ 1" forces the use of an intercept, "- 1" forces the use of no intercept
print(model.summary())

#Link to t-tests between different FSIQ and PIQ

data_fsiq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'}) 

data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})

data_long = pd.concat([data_fsiq, data_piq])

model = ols("iq ~ type", data_long).fit()

print(model.summary())

stats.ttest_ind(data['FSIQ'], data['PIQ'])

#Multiple Regression: including multiple factors

data = pd.read_csv('data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())

#Post-hox hypothesis testing: analysis of variance (ANOVA)

print(model.f_test([0,1,-1,0]))
