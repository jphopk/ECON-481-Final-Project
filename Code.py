#Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

#Data
df = pd.read_csv(
    '/Users/linzixuan/Documents/ECON-481-Final-Tester/Monthly_Macroeconomic_Factors.csv'
    ) # this should be whatever the path is to the file
df.set_index('Date', inplace=True) # set the date as the index

#Model
Y = df['House_Price_Index'] # Define dependent variable
Y = Y.to_numpy()
Y = Y.astype(float)
Y = Y.reshape(-1, 1)

X = df.drop('House_Price_Index', axis=1) # Define independent variables
X = sm.add_constant(X) # Add a constant to the independent values
X = X.to_numpy()

model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())