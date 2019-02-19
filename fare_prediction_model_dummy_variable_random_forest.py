import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import math
import pickle
from sklearn import linear_model
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer

print("start",dt.datetime.now())
#reading csv file

df= pd.read_csv('./fp_corporate_data.csv')

#print(df['faretype'].value_counts())
#sys.exit()
#remove last column which is unnamed
df = df[df.columns[:-1]]

#converting to datetime
df['pnr_date'] = pd.to_datetime(df.pnr_date).dt.date
df['departure_timestamp'] = pd.to_datetime(df['date_departure'] + ' ' + df['time_departure'] )


df = df.drop_duplicates(keep='first')

#remove Nan values
df = df.dropna(how='all')

#sort values
df = df.sort_values(['pnr_date'])

#orgin-dest pair
df['orgin-dest'] = df['origin_airport_code'] + ' ' + df['dest_airport_code'] 

#assigning category values
sectorCategoryList = pd.unique(df[['origin_airport_code', 'dest_airport_code']].values.ravel('K'))
df['origin_airport_code_label'] = df.origin_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes
df['dest_airport_code_label'] = df.dest_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes

departureDateTimePd = df['departure_timestamp'] 
df['departure_date'], df['departure_month'], df['departure_year'] = departureDateTimePd.dt.day, departureDateTimePd.dt.month, departureDateTimePd.dt.year
df['departure_hour'], df['departure_minute'], df['departure_day'] = departureDateTimePd.dt.hour, departureDateTimePd.dt.minute,departureDateTimePd.dt.weekday 

df['requested_timestamp'] = pnrDateTimeStamp = pd.to_datetime(df.pnr_date)
df['requested_date'], df['requested_month'], df['requested_year'] = pnrDateTimeStamp.dt.day, pnrDateTimeStamp.dt.month, pnrDateTimeStamp.dt.year

#days to departure
df['days_to_departure'] = (df['departure_timestamp']-df['requested_timestamp']).dt.days

#category values for airline_code
df['airline_code_label'] = df.airline_code.astype('category').cat.codes

#removing outliers
df = df.loc[(df['base_fare']>1000)&(df['base_fare']<10000)]

#df = df.loc[(df['departure_month']==1)|(df['departure_month']==2)|(df['departure_month']==3)|(df['departure_month']==4)|(df['departure_month']==5)]
print(df.shape)

aircodeDV = pd.get_dummies(df.airline_code,prefix="aircode")
originDV = pd.get_dummies(df.origin_airport_code,prefix="origin")
destDV = pd.get_dummies(df.dest_airport_code,prefix="destination")


df=pd.concat([df, aircodeDV],axis=1)
df=pd.concat([df, originDV],axis=1)
df=pd.concat([df, destDV],axis=1)

print("start",dt.datetime.now())
#df = pd.read_pickle('./final_corporate_converted_label_aircode_origin_variable.pkl')
#df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction_aircode.pkl')
#df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction_dummy_variable.pkl')
#df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction_dummy_variable.pkl')
#print(df['cabin_class'].value_counts())
#sys.exit()
#print(df.dtypes)
#sys.exit()
df = df.loc[(df['cabin_class']=='E')|(df['cabin_class']=='C')|(df['cabin_class']=='Y')]
df = df.loc[((df['origin_DEL']==1)&(df['destination_BOM']==1))]
#df = df.loc[(df['faretype']=='CF')|(df['faretype']=='R')]
#df['cabin_class_label'] = df.cabin_class.astype('category').cat.codes
#df['fare_type_label'] = df.faretype.astype('category').cat.codes
X = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'pnr_date',
					  'airline_code',
					  'flight_no',
					  'date_departure',
					  'departure_year',
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
					  'departure_minute',
				      'base_fare',
				      'origin_airport_code_label',
				      'dest_airport_code_label',
					'airline_code_label'
					  ])
print(X.dtypes)	     
#X.to_csv('/var/www/html/ML/flight_low_fare_prediction/data/low_fare_prediction_sample_data.csv')

#grad = GradientBoostingRegressor()
#X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.2, random_state=5)
#grad.fit(X_train, y_train)
#y_pred = grad.predict(X_test)

#mae = metrics.mean_absolute_error(y_pred,y_test)

#mse = metrics.mean_squared_error(y_pred,y_test)
#print("MSE",mse) 
#print("MAE",mae) 
#print("Score: ",grad.score(X_test,y_test))
#sys.exit(1)

#print(df.head())
regressor = RandomForestRegressor(n_estimators = 70,random_state=0)
y = df.base_fare;

"""
def rfr_model(X, y):
	# Perform Grid-Search
	gsc = GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': range(3,7),'n_estimators': (10, 50, 100),},cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
	X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.2, random_state=5)
	gsc.fit(X_train, y_train)
	dfGridSearch = pd.DataFrame(gsc.cv_results_)[['mean_test_score','std_test_score','params']]
	print(dfGridSearch.head())	
	dfGridSearch.to_csv('/var/www/html/gridSearchResult.csv')
	
	rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False,verbose=False)
	# Perform K-Fold CV
	scores = cross_val_predict(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')
"""
	
X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.2, random_state=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mae = metrics.mean_absolute_error(y_pred,y_test)
mse = metrics.mean_squared_error(y_pred,y_test)
print("MSE",mse) 
print("MAE",mae) 
print("Score: ",regressor.score(X_test,y_test))
pickle.dump(regressor, open('./final_pickle_pred.pkl', 'wb'))
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_test,y_test)
print(clf.score(X_test,y_test))
#X_test['actual'] = y_test
#X_test['pred'] = y_pred
#print(df.head(10))	
print("end",dt.datetime.now())

sys.exit(1)
data = X_test
data['actual_base_fare'] = y_test
data['predicted_base_fare'] = y_pred
print(data.head)
#pickle.dump(regressor, open('/var/www/html/ML/flight_low_fare_prediction/data/regressor30_model_object_test.pkl', 'wb'))
#df.to_pickle('/var/www/html/ML/flight_low_fare_prediction/data/model_result_graph.pkl')


