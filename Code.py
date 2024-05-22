#Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Data Cleaning & Presenting
df = pd.read_csv('file path')
df.set_index('Date', inplace=True) # set the date as the index
df

#Descriptive Statistics
df.describe() #descriptive statistics for each column

#Data Visualization for each variable depends on time
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
    plt.xticks([df.index.min(), df.index.max()]) #Remove the crowded x-axis labels but kept the first and last labels

plt.show()

#Data Visualization for house price index depends on different variables
for column in df.columns:
    if column != 'House_Price_Index':
        plt.scatter(df[column], df['House_Price_Index'])
        plt.xlabel(column)
        plt.ylabel('House Price Index')
        plt.title(f'Relationship between {column} and House Price Index')
        plt.show()

#Model with all variables
formula = 'House_Price_Index ~ ' + ' + '.join(
    df.columns.difference(['House_Price_Index'])) #get all the columns except for the house price index

model = smf.ols(formula, data=df)
results = model.fit()

print(results.summary())