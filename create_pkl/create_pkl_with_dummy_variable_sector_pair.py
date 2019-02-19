import pandas as pd
import sys

pd.set_option('display.max_columns', None)

#df = pd.read_pickle('./data/balmer_fare_prediction.pkl')

df = pd.read_pickle('./data/corporate_converted.pkl')


def getList(val):
	catdf1 = []
	for list in val:
		catdf1.append(list[0])
	return catdf1

catdf = pd.read_pickle('./data/category_aircode.pkl')
catdf = getList(catdf.values)
origindf = pd.read_pickle('./data/category_origin.pkl')
origindf = getList(origindf.values)
destdf = pd.read_pickle('./data/category_dest.pkl')
destdf = getList(destdf.values)

aircodeDV = pd.get_dummies(df.airline_code,prefix="aircode")
aircodeDV = aircodeDV.T.reindex(catdf).T.fillna(0)
originDV = pd.get_dummies(df.origin_airport_code,prefix="origin")
originDV = originDV.T.reindex(origindf).T.fillna(0)
destDV = pd.get_dummies(df.dest_airport_code,prefix="destination")
destDV = destDV.T.reindex(destdf).T.fillna(0)


df=pd.concat([df, aircodeDV],axis=1)
df=pd.concat([df, originDV],axis=1)
df=pd.concat([df, destDV],axis=1)

#sectorPair = pd.get_dummies(df.dest_airport_code,prefix="sector")
#df=pd.concat([df, sectorPair],axis=1)

df.to_pickle('./data/corporate_converted_dummy_variable_sector_pair.pkl')

