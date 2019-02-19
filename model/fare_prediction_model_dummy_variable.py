import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys

pd.set_option('display.max_columns', None)

df = pd.read_pickle('./data/fare_prediction_dummy_variable.pkl')

X = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'pnr_date',
					  'airline_code',
					  'flight_no',
					  'date_departure',
					  'time_departure',
					  'cabin_class',
					  'trip_type',
					  'faretype',
					  'fare_basis_code',
					  'departure_timestamp',
					  'orgin-dest',
					  'request_date',
					  'requested_timestamp',
					  'days_to_departure',
					  'origin_airport_code_label',
					  'dest_airport_code_label',
					  'airline_code_label',
					  'base_fare'])	
			
				  
y = df['base_fare'] 

#df = df.loc[(df['airline_code']=='6E')]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)
#-----------------------------------------------------------LINEAR REGRESSION----------------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 	# 35119461.09640596 
print('Score:',linreg.score(X_test,y_test))		#-10610479489412.72


#------------------------------------------------------------RANDOM FOREST ------------------------------------------------------------------------

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Score:',regressor.score(X_test,y_test))	
"""

