import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import Normalizer 


df = pd.read_pickle('/var/www/html/ML/git_low_fare_prediction/data/corporate_converted_dummy_variable.pkl')


df = df.drop(columns=['origin_airport_code',
					  'dest_airport_code',
					  'pnr_date',
					  'airline_code',
					  'flight_no',
					  'date_departure',
					  'time_departure',
					  'Cabin class',
					  'Trip type',
					  'Fare type',
					  'departure_timestamp',
					  'orgin-dest',
					  'requested_timestamp',
					  'days_to_depature',
					  'origin_airport_code_label',
					  'dest_airport_code_label',
					  'airline_code_label'])		 
X = df.drop(columns=['base_fare'])
y = df['base_fare'] 
nz = Normalizer()
X = nz.fit(X).transform(X)
#df = df.loc[(df['airline_code']=='6E')]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=4)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
print('R2_score:', (metrics.r2_score(y_test, y_pred)))
print('Score:',linreg.score(X_test,y_test))



