# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
# load data into a Pandas DataFrame
df = pd.read_csv(r'小论文相关\final_data_O3.csv')

# Split the data into features and target variables
X_O =df.drop(['AQI','O3','date','hour'], axis=1)
y_ozone = df['O3']

# Split the data into training and testing sets
X_Otrain, X_Otest, y_train_ozone, y_test_ozone = train_test_split(X_O, y_ozone, test_size=0.2)


# train the random forest models
rf_ozone = RandomForestRegressor(n_estimators=120,max_depth=16,min_samples_split=2,
min_samples_leaf=2,min_weight_fraction_leaf=0.01,max_features=0.9)
rf_ozone.fit(X_O, y_ozone)

# Predict on the test set
ozone_pred = rf_ozone.predict(X_Otest)

# Evaluate the performance of the model
mse_ozone = mean_squared_error(y_test_ozone, ozone_pred)

print(f"Mean Squared Error for Ozone: {mse_ozone:.4f}")

# calculate feature importance
importance_ozone = rf_ozone.feature_importances_

#importance ranking
def impsort(arr):
    n=len(arr)
    for x in range(n-1):
        for y in range(n-1-x):
            if arr[y]>arr[y+1]:
                arr[y],arr[y+1]=arr[y+1],arr[y]
    arr=arr[::-1]
    return arr
importance_ozone_sorted=impsort(importance_ozone)
print(importance_ozone_sorted)

# visualize the feature importance
plt.figure(dpi=300)
features_O = X_O.columns
plt.bar(features_O, importance_ozone_sorted)
plt.xticks(rotation=45)
plt.title('Variable Importance for O3')
plt.show()
