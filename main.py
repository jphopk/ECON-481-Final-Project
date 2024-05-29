#Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score


#Data Tables
df = pd.read_csv('Monthly_Macroeconomic_Factors.csv') #it should be a path in your computer

# set the date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

display(df)
display(df.describe())


#Data Work Before Visualizations

#standardize CPI to the same year as HPI (January 1, 2000)
df['Consumer_Price_Index'] /= df.loc['2000-01-01', 'Consumer_Price_Index']
df['Consumer_Price_Index'] *= 100

#plots of every variable verus time
variables = df.columns
plt.figure(figsize=(15, 30))  
for i, var in enumerate(variables, start=1):
    plt.subplot(len(variables), 1, i)
    plt.plot(df.index, df[var])
    plt.title(var)
    plt.xlabel('Time')
    plt.ylabel(var)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()


#Comparative Graphs

#Plot originial HPI (including inflation bias) against CPI
plt.figure(figsize=(12, 6))

# Plot the first dataset
plt.plot(df.index, df['House_Price_Index'], label='House Price Index', color='blue')

# Plot the second dataset on the same axes
plt.plot(df.index, df['Consumer_Price_Index'], label='Consumer Price Index', color='red')

# Add titles and labels
plt.title('Superimposed Plots of HPI and CPI')
plt.xlabel('Time')
plt.ylabel('HPI and CPI')

# Add a legend
plt.legend()

#divide HPI by CPI to see HPI without inflation bias
df['Adj_House_Price_Index'] = df['House_Price_Index']/ df['Consumer_Price_Index']
df['Adj_House_Price_Index'] *= 100

#Plot Adj_HPI versus time and CPI versus time to see if Adj_HPI is growing faster than CPI/inflation

plt.figure(figsize=(12, 6))

# Plot the first dataset
plt.plot(df.index, df['Adj_House_Price_Index'], label='Adjusted House Price Index', color='blue')

# Plot the second dataset on the same axes
plt.plot(df.index, df['Consumer_Price_Index'], label='Consumer Price Index', color='red')

# Add titles and labels
plt.title('Superimposed Plots of Adjusted HPI and CPI')
plt.xlabel('Time')
plt.ylabel('Adj HPI and CPI')

# Add a legend
plt.legend()

plt.show()


#Computations of Actual Real GDP & its Visualization

# changing real gdp from percentage change to actual real gdp, but it will be relative
# to a base year since we do not have the data of actual dollar amount of rgdp
# we will use January 1, 2000 as the base year as before

# pretending rgdp at index -1 is "1" just for simplicity. now calculate real gdp at each date using the percent
# change for that date from the date before. we need to divide each rgdp data point by 12 to change from annual to 
# monthly form

df['Actual_RGDP'] = df['Real_GDP'] / 12

# convert to percents
df['Actual_RGDP'] /= 100

#add 1 to first date to pretend rgdp in index -1 is 1
df.loc['1987-01-01', 'Actual_RGDP'] += 1


#change each Actual_RGDP element to prior element multiplied by (1+current element)

for i in range(1, len(df)):
    df.iloc[i, df.columns.get_loc('Actual_RGDP')] = df.iloc[i-1, df.columns.get_loc('Actual_RGDP')] * (1 + df.iloc[i, df.columns.get_loc('Actual_RGDP')])

#standardize Actual_RGDP to the same year as HPI (January 1, 2000)
df['Actual_RGDP'] /= df.loc['2000-01-01', 'Actual_RGDP']
df['Actual_RGDP'] *= 100

#Plot Actual_RGDP versus time
plt.figure(figsize=(12, 6))  
plt.plot(df.index, df['Actual_RGDP'])
plt.title('RGDP Index Versus Time')
plt.xlabel('Time')
plt.ylabel('RGDP Index')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


#Graph of actual real GDP vs Adjusted HPI over time

#Superimpose RGDP index versus time over Adjusted HPI versus time. (Both have same base date for comparing purposes)
plt.figure(figsize=(12, 6))

# Plot the first dataset
plt.plot(df.index, df['Adj_House_Price_Index'], label='Adjusted House Price Index', color='blue')

# Plot the second dataset on the same axes
plt.plot(df.index, df['Actual_RGDP'], label='Real GDP Index', color='red')

# Add titles and labels
plt.title('Superimposed Plots of Adj HPI and RGDP Index')
plt.xlabel('Time')
plt.ylabel('HPI and RGDP Index')

# Add a legend
plt.legend()

# Show the plot
plt.show()


#Cross Validation & Feature Selection

# Add a constant to the df for the VIF calculation
X_vif = df.drop(columns=['House_Price_Index']).assign(const=1)

# Calculate and print VIF
vif = pd.DataFrame()
vif["variables"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)


#Optimal Alpha & Lasso Model

# Define dependent and independent Variables
X = df[['Unemployment_Rate', 'Real_GDP', 'Mortgage_Rate']]
y = df['House_Price_Index']

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


# Accuarcy Metrics
feature_columns = df[['Unemployment_Rate', 'Real_GDP', 'Mortgage_Rate']]# List of feature column names, without House_Price_Index
target_column = df['House_Price_Index']  # Name of the target column

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_columns, target_column, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = Lasso(alpha=lasso_cv.alpha_)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)