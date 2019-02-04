import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#df = pd.read_pickle('../data/corporate_converted_dummy_variable.pkl')
df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction.pkl')

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
					  'departure_timestamp',
					  'orgin-dest',
					  'request_date',
					  'fare_basis_code',
					  'requested_timestamp',
					  'days_to_departure'
					  ])	
#X = df.drop(['origin_airport_code','dest_airport_code','pnr_date','airline_code','flight_no','date_departure','time_departure','base_fare','Cabin class','Trip type','Fare type','departure_timestamp','orgin-dest','requested_timestamp','origin_airport_code_label','dest_airport_code_label','airline_code_label'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.33, random_state=5)
regressor = RandomForestRegressor(n_estimators=70, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = metrics.mean_absolute_error(y_pred,y_test)
print("MAE",mse) 	# 0.0057394245787786
print("Score: ",regressor.score(X_test,y_test))	# 0.9999999993382981
