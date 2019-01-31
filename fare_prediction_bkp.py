import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#df = pd.read_pickle('/home/infiniti/Desktop/ML/pandas_corporate.pkl')
df = pd.read_pickle('/home/infiniti/Desktop/ML/pandas_corporate_airline_code.pkl')

"""X = df[['requested_date',
		  'requested_month',
		  'requested_year',
		  'airline_code', 
		  'origin_airport_code', 
		  'dest_airport_code', 
		  'num_passenger',
		  'departure_hour',
		  'departure_minute',
		  'departure_date',
		  'departure_month',
		  'departure_year'
		 ]]
		 """
X = df.drop(columns=['base_fare'])		 
y = df['base_fare'] 
print(df.dtypes)
"""
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
#print(linreg.score(X_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
y_pred = pd.DataFrame({'pred':y_pred.tolist()})
#print(y_test.head(20))
#print(y_pred.head(20))
print(linreg.score(X_test,y_test))
df1 = pd.concat([y_test,y_pred],axis=1)
print(df1.head())
"""
