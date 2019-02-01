import pandas as pd
import datetime as dt

#reading csv file
df_corporate = pd.read_csv('/var/www/html/ML/flight_low_fare_prediction/csv_data/fp_corporate_data.csv')
df_service = pd.read_csv('/var/www/html/ML/flight_low_fare_prediction/csv_data/fp_service_data.csv')

#df_corporate = df_corporate.head(5)
#df_service = df_service.head(5)

#remove last column which is unnamed
df_corporate = df_corporate[df_corporate.columns[:-1]]

#converting to datetime
df_corporate['pnr_date'] = pd.to_datetime(df_corporate.pnr_date).dt.date
df_corporate['departure_timestamp'] = pd.to_datetime(df_corporate['date_departure'] + ' ' + df_corporate['time_departure'] )

df_service['pnr_date'] = pd.to_datetime(df_service.pnr_date).dt.date
df_service['departure_timestamp'] = pd.to_datetime(df_service['date_departure'] + ' ' + df_service['time_departure'] )

pd.set_option('display.max_columns', None)
df1 = pd.concat([df_corporate,df_service],sort=False)
df1 = df1.drop_duplicates(keep='first')

#remove Nan values
df1 = df1.dropna(how='all')


#sort values
df1 = df1.sort_values(['pnr_date'])

#orgin-dest pair
df1['orgin-dest'] = df1['origin_airport_code'] + ' ' + df1['dest_airport_code'] 

#assigning category values
sectorCategoryList = pd.unique(df1[['origin_airport_code', 'dest_airport_code']].values.ravel('K'))
df1['origin_airport_code_label'] = df1.origin_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes
df1['dest_airport_code_label'] = df1.dest_airport_code.astype('category', CategoricalDtype=sectorCategoryList).cat.codes

departureDateTimePd = df1['departure_timestamp'] 
df1['departure_date'], df1['departure_month'], df1['departure_year'] = departureDateTimePd.dt.day, departureDateTimePd.dt.month, departureDateTimePd.dt.year
df1['departure_hour'], df1['departure_minute'] = departureDateTimePd.dt.hour, departureDateTimePd.dt.minute 

df1['requested_timestamp'] = pnrDateTimeStamp = pd.to_datetime(df1.pnr_date)
df1['requested_date'], df1['requested_month'], df1['requested_year'] = pnrDateTimeStamp.dt.day, pnrDateTimeStamp.dt.month, pnrDateTimeStamp.dt.year

#days to departure
df1['days_to_depature'] = (df1['departure_timestamp']-df1['requested_timestamp']).dt.days

#category values for airline_code
df1['airline_code_label'] = df1.airline_code.astype('category').cat.codes


df = df1[['requested_date',
		  'requested_month',
		  'requested_year',
		  'requested_timestamp',
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


#for airline code get_dummies
newdf = pd.get_dummies(df.airline_code,prefix="aircode_")
df=pd.concat([df, newdf],axis=1)
df = df.drop(columns=['airline_code']) 

#getting only indigo airline
#df = df.loc[(df['airline_code']=='6E')]

print(df.head())
print(df.dtypes)
print(df.shape)
#df.to_pickle('/var/www/html/ML/flight_low_fare_prediction/fare_prediction.pkl')
df.to_csv('/var/www/html/ML/flight_low_fare_prediction/fare_prediction.csv')

