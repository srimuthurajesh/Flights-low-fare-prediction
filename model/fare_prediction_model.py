#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys
pd.set_option('display.max_columns', None)

df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction_aircode.pkl')
df = df.drop(columns=['airline_code_label'])
#df = pd.read_pickle('../data/fare_prediction.pkl')
					  
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
print(X.head())

#df = df.loc[(df['airline_code']=='6E')]	#this makes no impact

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  	
print('score:',linreg.score(X_test,y_test))		

#without dummy airlinecode dummy,both db		1555.7366092302323 	#0.037622683295648995
#with dummy airlinecode dummy,both db 		1539.5643475868835  #0.05630064478947449

#without dummy airlinecode dummy,corporate db		1504.6962136562636	 #0.008385139620986637
#with dummy airlinecode dummy,corporate db 		85827374.42618504 #-396621564244124.0
 	


