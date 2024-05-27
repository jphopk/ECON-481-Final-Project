import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Data
df = pd.read_csv('/Users/sdesai/Desktop/econ-481/ECON-481-Final-Project/Monthly_Macroeconomic_Factors.csv')
df.set_index('Date', inplace=True) # set the date as the index

# Assuming you have a DataFrame named 'df' containing your data
# You should replace 'feature_columns' with the actual columns you want to use as features
feature_columns = df.columns.tolist()[1:]  # List of feature column names, without House_Price_Index
target_column = 'House_Price_Index'  # Name of the target column

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
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
