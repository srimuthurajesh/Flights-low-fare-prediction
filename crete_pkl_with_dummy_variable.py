import pandas as pd

df = pd.read_pickle('/var/www/html/ML/git_low_fare_prediction/data/corporate_converted.pkl')
aircodeDV = pd.get_dummies(df.airline_code,prefix="aircode_")
originAircodeDV = pd.get_dummies(df.origin_airport_code,prefix="origin_")
destAircodeDV = pd.get_dummies(df.dest_airport_code,prefix="destination_")

df=pd.concat([df, aircodeDV],axis=1)
df=pd.concat([df, originAircodeDV],axis=1)
df=pd.concat([df, originAircodeDV],axis=1)
df.to_pickle('/var/www/html/ML/git_low_fare_prediction/data/corporate_converted_dummy_variable.pkl')

