import pandas as pd

#catdf = pd.read_pickle('./data/category_test.pkl')
df = pd.read_pickle('./data/fare_prediction.pkl')
"""
catdf1 = []
for list in catdf.values:
    for x in list:
        catdf1.append(x)


dummies = pd.get_dummies(df.airline_code, prefix='aircode')
dummies = dummies.T.reindex(catdf).T.fillna(0)

"""
aircodeDV = pd.get_dummies(df.airline_code,prefix="aircode")
originAircodeDV = pd.get_dummies(df.origin_airport_code,prefix="origin")
destAircodeDV = pd.get_dummies(df.dest_airport_code,prefix="dest")

dfAircodeDV = pd.DataFrame({'category_aircode':aircodeDV.columns})
dfOriginDV = pd.DataFrame({'category_origin':originAircodeDV.columns})
dfDestDV = pd.DataFrame({'category_dest':destAircodeDV.columns})
print(dfAircodeDV.head())
print(dfOriginDV.head())
dfAircodeDV.to_pickle('./data/category_aircode.pkl')
dfOriginDV.to_pickle('./data/category_origin.pkl')
dfDestDV.to_pickle('./data/category_dest.pkl')
