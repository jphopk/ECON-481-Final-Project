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

#plots
variables = df.columns
plt.figure(figsize=(15, 30))  # Increase the figure size to 15 inches wide by 30 inches tall
for i, var in enumerate(variables, start=1):
    plt.subplot(len(variables), 1, i)
    plt.plot(df.index, df[var])
    plt.title(var)
    plt.xlabel('Time')
    plt.ylabel(var)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

#lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add a constant to the df for the VIF calculation
X_vif = df.drop(columns=['House_Price_Index']).assign(const=1)

# Calculate and print VIF
vif = pd.DataFrame()
vif["variables"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)

# Assume df is your DataFrame and 'House_Price_Index' is your target variable
X = df[['Unemployment_Rate', 'Real_GDP', 'Mortgage_Rate']]
y = df['House_Price_Index']

from sklearn.linear_model import LassoCV

# Create a LassoCV object
lasso_cv = LassoCV(cv=5)

# Fit it to the data
lasso_cv.fit(X, y)

# The optimal alpha level is stored in `lasso_cv.alpha_`
print('Optimal alpha level:', lasso_cv.alpha_)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Lasso regression model
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)

# Create a DataFrame with the coefficients and column names
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})

print(coef_df)