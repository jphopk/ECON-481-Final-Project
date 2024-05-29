import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Data
df = pd.read_csv('ENTER_PATH_HERE')
df.set_index('Date', inplace=True) # set the date as the index

#Model with all variables
formula = 'House_Price_Index ~ ' + ' + '.join(
    df.columns.difference(['House_Price_Index'])) #get all the columns except for the house price index

model = smf.ols(formula, data=df)
results = model.fit()
print(results.summary())

X = df.drop('House_Price_Index', axis=1)
y = df['House_Price_Index']

X = sm.add_constant(X)

import matplotlib.pyplot as plt
ser = results.params
for i in range(1, len(ser)) :
    x = X[X.columns[i]]
    y = y
    
    plt.scatter(x, y, s=20)
    m = ser.iloc[i]
    y = m * x
    plt.plot(x, y)
    plt.xlabel(ser.index[i])
    plt.ylabel('House_Price_Index')
    plt.title('Scatter Plot of X vs House_Price_Index')
    plt.show()