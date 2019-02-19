import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import math
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#df = pd.read_pickle('./data/balmer_fare_prediction_dummy_variable.pkl')
#print(df.shape)
df = pd.read_pickle('./data/corporate_converted.pkl')
count = pd.unique(df['orgin-dest'].values.ravel('K'))
sectorPair = pd.get_dummies(df['orgin-dest'],prefix="sector")
df=pd.concat([df, sectorPair],axis=1)

#df = pd.read_pickle('./data/fare_prediction_aircode.pkl')
#df = pd.read_pickle('./data/fare_prediction_dummy_variable.pkl')
#df = pd.read_pickle('./data/fare_prediction_dummy_variable.pkl')

X = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'origin_airport_code_label',
					  'dest_airport_code_label',
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
					  'requested																																																																																																																																																																																																																														_date',
					  'fare_basis_code',
					  'requested_timestamp',
					  'days_to_departure',
					  'airline_code_label',
				      'base_fare'
					  ])
"""
X = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'origin_airport_code_label',
					  'dest_airport_code_label',
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
					  'days_to_departure',
					  'airline_code_label',
				     ])
"""				     
#X.to_csv('./data/low_fare_prediction_sample_data.csv')
#sys.exit()		

"""
X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.2, random_state=5)
regressor = RandomForestRegressor(n_estimators=30,random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
"""
gradient = GradientBoostingRegressor()
gradient.fit(X_train,y_train)

mse = metrics.mean_absolute_error(y_pred,y_test)
print("MAE",mse) 
print("Score: ",regressor.score(X_test,y_test))	
sys.exit()
data = X_test
data['actual_base_fare'] = y_test
data['predicted_base_fare'] = y_pred
#print(data.head)
#pickle.dump(regressor, open('../data/regressor30_model_object_test.pkl', 'wb'))
df.to_pickle('./data/model_result_graph.pkl')


