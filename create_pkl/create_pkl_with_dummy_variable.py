import pandas as pd

df = pd.read_pickle('./data/corporate_converted.pkl')
aircodeDV = pd.get_dummies(df.airline_code,prefix="aircode")
originAircodeDV = pd.get_dummies(df.origin_airport_code,prefix="origin")
destAircodeDV = pd.get_dummies(df.dest_airport_code,prefix="destination")

df=pd.concat([df, aircodeDV],axis=1)
df=pd.concat([df, originAircodeDV],axis=1)
df=pd.concat([df, destAircodeDV],axis=1)
df.to_pickle('./data/corporate_converted_dummy_variable.pkl')

