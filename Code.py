#Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

#Data
df = pd.read_csv('file path')
df.set_index('Date', inplace=True) # set the date as the index

#Model
formula = 'House_Price_Index ~ ' + ' + '.join(
    df.columns.difference(['House_Price_Index'])) #get all the columns except for the house price index

model = smf.ols(formula, data=df)
results = model.fit()

print(results.summary())