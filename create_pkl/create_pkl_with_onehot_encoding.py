import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#To display all the columns and rows
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

df = pd.read_pickle('./data/pandas_without_encoding.pkl')
print(df.head())

# Perform one hot encoding for airlines code.
airline_code = OneHotEncoder()
X = airline_code.fit_transform(df.airline_code.values.reshape(-1,1)).toarray()
#print(X)
dfOneHot = pd.DataFrame(X, columns = ["airline_code_"+str(int(i)) for i in range(X.shape[1])])
df = df.loc[~df.index.duplicated(keep='first')]
dfOneHot = dfOneHot.loc[~dfOneHot.index.duplicated(keep='first')]
df = pd.concat([df, dfOneHot], axis=1)
df = df.dropna()

# Perform one hot encoding for origin airport code.
origin_airport_code = OneHotEncoder()
X = origin_airport_code.fit_transform(df.origin_airport_code.values.reshape(-1,1)).toarray()
#print(X)
dfOneHot = pd.DataFrame(X, columns = ["origin_airport_code"+str(int(i)) for i in range(X.shape[1])])
df = df.loc[~df.index.duplicated(keep='first')]
dfOneHot = dfOneHot.loc[~dfOneHot.index.duplicated(keep='first')]
df = pd.concat([df, dfOneHot], axis=1)
df = df.dropna()

# Perform one hot encoding for destination airport code.

origin_airport_code = OneHotEncoder()
X = origin_airport_code.fit_transform(df.origin_airport_code.values.reshape(-1,1)).toarray()
#print(X)
dfOneHot = pd.DataFrame(X, columns = ["dest_airport_code"+str(int(i)) for i in range(X.shape[1])])
df = df.loc[~df.index.duplicated(keep='first')]
dfOneHot = dfOneHot.loc[~dfOneHot.index.duplicated(keep='first')]
df = pd.concat([df, dfOneHot], axis=1)
df = df.dropna()

#After encoded all the feature varaible write a new pickle file
df.to_pickle('./data/pandas_airline_sector_encoded.pkl')
