import pandas as pd
import datetime as dt
import sys
pd.set_option('display.max_columns', None)


#reading csv file
df_corporate = pd.read_csv('../csv_data/fp_corporate_data.csv')
df_service = pd.read_csv('../csv_data/fp_service_data.csv')
df_service.truncate()
#remove last column which is unnamed
df_corporate = df_corporate[df_corporate.columns[:-1]]

#converting to datetime
df_corporate['pnr_date'] = pd.to_datetime(df_corporate.pnr_date).dt.date
df_corporate['departure_timestamp'] = pd.to_datetime(df_corporate['date_departure'] + ' ' + df_corporate['time_departure'] )

df_service['pnr_date'] = pd.to_datetime(df_service.pnr_date).dt.date
df_service['departure_timestamp'] = pd.to_datetime(df_service['date_departure'] + ' ' + df_service['time_departure'] )

df = pd.concat([df_corporate,df_service],sort=False)
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
df['departure_hour'], df['departure_minute'] = departureDateTimePd.dt.hour, departureDateTimePd.dt.minute 

df['requested_timestamp'] = pnrDateTimeStamp = pd.to_datetime(df.pnr_date)
df['requested_date'], df['requested_month'], df['requested_year'] = pnrDateTimeStamp.dt.day, pnrDateTimeStamp.dt.month, pnrDateTimeStamp.dt.year

#days to departure
df['days_to_departure'] = (df['departure_timestamp']-df['requested_timestamp']).dt.days

#category values for airline_code
df['airline_code_label'] = df.airline_code.astype('category').cat.codes


"""
df = df[['requested_date',
		  'requested_month',
		  'requested_year',
		  'requested_timestamp',
		  'days_to_departure',
		  'airline_code',
		  'airline_code_label', 
		  'origin_airport_code',
		  'origin_airport_code_label',
		  'dest_airport_code',
		  'dest_airport_code_label', 
		  'orgin-dest', 
		  'num_passenger',
		  'departure_hour',
		  'departure_minute',
		  'departure_date',
		  'departure_month',
		  'departure_year',
		  'departure_timestamp',
		  'base_fare'
		  ]]
"""


#removing outliers
df = df.loc[(df['base_fare']>1000)&(df['base_fare']<10000)]

#To create dataframe without airline code bucketing
df.to_pickle('../data/corporate_fare_prediction.pkl')

"""
#for airline code get_dummies
newdf = pd.get_dummies(df.airline_code,prefix="aircode")
df=pd.concat([df, newdf],axis=1)
df.to_pickle('../data/corporate_fare_prediction_aircode.pkl')
"""
print(df.head())
print(df.dtypes)
print(df.shape)
