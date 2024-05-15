#Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Data
df = pd.read_csv('file path')
df.set_index('Date', inplace=True) # set the date as the index

#Model with all variables
formula = 'House_Price_Index ~ ' + ' + '.join(
    df.columns.difference(['House_Price_Index'])) #get all the columns except for the house price index

model = smf.ols(formula, data=df)
results = model.fit()

print(results.summary())

#Check for autocorrelation, heteroskedasticity, and multicollinearity of our model
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.drop('House_Price_Index', axis=1)
y = df['House_Price_Index']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

residuals = model.resid

dw_result = durbin_watson(residuals) # Apply Durbin-Watson test on the residuals
print(f'Durbin-Watson statistic: {dw_result}')

bp_test = het_breuschpagan(residuals, model.model.exog) # Apply Breusch-Pagan test
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # Calculate VIF for each explanatory variable
print(vif)

#It appears there are all three issues present in our model. We will address them one by one.