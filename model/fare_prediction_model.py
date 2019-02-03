import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys

df = pd.read_pickle('./data/fare_prediction.pkl')
print(df.columns)
sys.exit()
X = df[['requested_date',
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
		 
#X = df.drop(columns=['base_fare'])		 
y = df['base_fare'] 

#df['orgin-dest'] = df['origin_airport_code'] + ' ' + df['dest_airport_code'] 

df = df.loc[(df['airline_code']=='6E')]

print(df.shape)
"""
print(df.dtypes)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('score:',linreg.score(X_test,y_test))
"""
