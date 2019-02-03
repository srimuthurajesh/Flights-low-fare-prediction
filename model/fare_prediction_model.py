#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys
pd.set_option('display.max_columns', None)

#df = pd.read_pickle('../data/fare_prediction_aircode.pkl')
#df = df.drop(columns=['origin_airport_code_label','dest_airport_code_label','airline_code_label'])

df = pd.read_pickle('../data/fare_prediction.pkl')
					  
X = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'pnr_date',
					  'flight_no',
					  'date_departure',
					  'time_departure',
					  'cabin_class',
					  'trip_type',
					  'airline_code',
					  'faretype',
					  'fare_basis_code',
					  'departure_timestamp',
					  'orgin-dest',
					  'request_date',
					  'requested_timestamp',
					  'days_to_departure',
					  'base_fare'])	
y = df['base_fare'] 

#df = df.loc[(df['airline_code']=='6E')]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  	
print('score:',linreg.score(X_test,y_test))		

#without airlinecode dummy		1555.7366092302323 	#0.037622683295648995
#with airlinecode dummy 		1539.5643475868835  #0.05630064478947449


