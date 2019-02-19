import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
import pickle
import math
from sklearn.linear_model import Ridge
pd.set_option('display.max_columns', None)

#df = pd.read_pickle('../pandas_corporate_airline_code.pkl')
df = pd.read_pickle('/var/www/html/ML/flight_low_fare_prediction/data/fare_prediction.pkl')

print(stats.kstest(df.base_fare,'t',(10,)))
