import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_pickle('./corporate_converted_dummy_variable.pkl')
#print(df.head().dtypes)
#sys.exit(1)
X = df.drop(['origin_airport_code','dest_airport_code','pnr_date','airline_code','flight_no','date_departure','time_departure','base_fare','Cabin class','Trip type','Fare type','departure_timestamp','orgin-dest','requested_timestamp','origin_airport_code_label','dest_airport_code_label','airline_code_label'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X, df.base_fare, test_size=0.33, random_state=5)
regressor = RandomForestRegressor(n_estimators=70, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)
print("mse",mse) #2868604.735152483
print("mae",mae) #1396.9930528124487
print(math.sqrt(mse)) #1693.6955851487844
print("test score",regressor.score(X_test,y_test))
