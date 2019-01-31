import pandas as pd
import datetime as dt

#reading csv file
df_corporate = pd.read_csv('/home/infiniti/Desktop/ML/fp_corporate_data.csv')
df_service = pd.read_csv('/home/infiniti/Desktop/ML/fp_service_data.csv')

#remove last column which is unnamed
df_corporate = df_corporate[df_corporate.columns[:-1]]

#converting to datetime
df_corporate['pnr_date'] = requestDatePd = pd.to_datetime(df_corporate.pnr_date).dt.date
df_corporate['departure_timestamp'] = pd.to_datetime(df_corporate['date_departure'] + ' ' + df_corporate['time_departure'] )

df_service['pnr_date'] = requestDatePd = pd.to_datetime(df_service.pnr_date).dt.date
df_service['departure_timestamp'] = pd.to_datetime(df_service['date_departure'] + ' ' + df_service['time_departure'] )

pd.set_option('display.max_columns', None)
df1 = pd.concat([df_corporate,df_service],sort=False)
df1 = df1.drop_duplicates(keep='first')

#remove Nan values
df1 = df1.dropna(how='all')


#sort values
df1 = df1.sort_values(['pnr_date'])

#assigning category values
sectorCategoryList = pd.unique(df1[['origin_airport_code', 'dest_airport_code']].values.ravel('K'))
df1['origin_airport_code'] = df1.origin_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes
df1['dest_airport_code'] = df1.dest_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes

departureDateTimePd = df1['departure_timestamp'] 
df1['departure_date'], df1['departure_month'], df1['departure_year'] = departureDateTimePd.dt.day, departureDateTimePd.dt.month, departureDateTimePd.dt.year
df1['departure_hour'], df1['departure_minute'] = departureDateTimePd.dt.hour, departureDateTimePd.dt.minute 

pnrDateTimeStamp = pd.to_datetime(df1.pnr_date)
df1['requested_date'], df1['requested_month'], df1['requested_year'] = pnrDateTimeStamp.dt.day, pnrDateTimeStamp.dt.month, pnrDateTimeStamp.dt.year

df = df1[['requested_date',
		  'requested_month',
		  'requested_year',
		  'airline_code', 
		  'origin_airport_code', 
		  'dest_airport_code', 
		  'num_passenger',
		  'departure_hour',
		  'departure_minute',
		  'departure_date',
		  'departure_month',
		  'departure_year',
		  'base_fare'
		  ]]
#assigning category values
#df1['airline_code'] = df1.airline_code.astype('category').cat.codes
newdf = pd.get_dummies(df.airline_code,prefix="aircode_")
df=pd.concat([df, newdf],axis=1)
df = df.drop(columns=['airline_code'])		 

print(df.head())
print(df.shape)
df.to_pickle('/home/infiniti/Desktop/ML/pandas_corporate_airline_code.pkl')
#df.to_csv('/home/infiniti/Desktop/ML/pandas_corporate.csv')
#df.to_csv('/home/infiniti/Desktop/ML/pandas_service.csv')

