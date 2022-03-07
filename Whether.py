#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 


# In[2]:


df=pd.read_csv("/Users/chauhanabhishek/Downloads/Weather_data.csv")


# In[3]:


df.head()


# In[4]:


df[" _conds"].value_counts()


# # Part 1: A quick analysis of weather in delhi.

# In[5]:


plt.figure(figsize=(15,10))
df[" _conds"].value_counts().head(15).plot(kind="bar")
plt.title("15 most common weathers in delhi")
plt.show()


# It is clearly visible that haze and smoke are most common weathers in delhi

# In[6]:


plt.figure(figsize=(15, 10))
plt.title("Common wind direction in delhi")
df[' _wdire'].value_counts().plot(kind="bar")
plt.plot()


# North and West are the most common wind directions in dehi.

# In[7]:


plt.figure(figsize=(15, 10))
sns.distplot(df[' _tempm'],bins=[i for i in range(0,61,5)], kde=False)
plt.title("Distribution of Temperatures")
plt.grid()
plt.show()


# Most common temperature scale in Delhi is from 25 to 35 degree.

# In[8]:


df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])


# In[9]:


df['datetime_utc'] 


# In[10]:


df.isna().sum()


# In[11]:


# imputing the missing value in temperatre feature with mean.
df[' _tempm'].fillna(df[' _tempm'].mean(), inplace=True)


# In[12]:


df[' _tempm'].isna().sum()
# filled all missing values with mean()


# In[13]:


str(df['datetime_utc'][0])


# In[14]:


# a function to extract year part from the whole date
def get_year(x):
  return x[0:4]


# In[15]:


# a function to extract month part from the whole date
def get_month(x):
  return x[5:7]


# In[16]:


df['year'] = df['datetime_utc'].apply(lambda x: get_year(str(x)))
df['month'] = df['datetime_utc'].apply(lambda x: get_month(str(x)))


# In[17]:


df['year']


# In[18]:


temp_year = pd.crosstab(df['year'], df['month'], values=df[' _tempm'], aggfunc='mean')


# In[19]:


plt.figure(figsize=(15, 10))
sns.heatmap(temp_year, cmap='coolwarm', annot=True)
plt.title("Average Tempearture in Delhi from 1996 to 2017")
plt.show()


# In[20]:


df[' _hum'].isna().sum()


# In[21]:


# imputing missing values in _hum feature with mean
df[' _hum'].fillna(df[' _hum'].mean(), inplace=True)


# In[22]:


humidity_year = pd.crosstab(df['year'], df['month'], values=df[' _hum'], aggfunc='mean')


# In[23]:


plt.figure(figsize=(15, 10))
sns.heatmap(humidity_year, cmap='coolwarm', annot=True)
plt.title("Average Humidity in Delhi from 1996 to 2017")
plt.show()


# # TimeSeries Forecasting
# 

# In[24]:


# taking only temperature feature as values and datetime feature as index in the dataframe for time series forecasting of temperature
data = pd.DataFrame(list(df[' _tempm']), index=df['datetime_utc'], columns=['temp'])


# In[25]:


data


# In[26]:


data = data.resample('D').mean()


# In[27]:



data.temp.isna().sum()


# In[28]:


data.fillna(data['temp'].mean(), inplace=True)


# In[29]:


data.temp.isna().sum()


# In[30]:


data.shape


# In[31]:


data


# In[32]:


plt.figure(figsize=(25, 7))
plt.plot(data, linewidth=.5)
plt.grid()
plt.title("Time Series (Years vs Temp.)")
plt.show()


# In[33]:


# Scaling data to get rid of outliers
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
data_scaled = scalar.fit_transform(data)


# In[34]:


data_scaled


# In[35]:


data_scaled.shape


# In[36]:


timestep = 30
X= []
Y=[]
for i in range(len(data_scaled)- (timestep)):
    X.append(data_scaled[i:i+timestep])
    Y.append(data_scaled[i+timestep])


# In[37]:


X=np.asanyarray(X)
Y=np.asanyarray(Y)


# In[38]:


k = 7200
Xtrain = X[:k,:,:]
Xtest = X[k:,:,:]    
Ytrain = Y[:k]    
Ytest= Y[k:] 


# In[39]:


X.shape


# In[40]:


Xtrain.shape


# In[41]:


Xtest.shape


# In[42]:


from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential


# In[43]:


from tensorflow.keras.layers import Bidirectional, Dropout


# In[44]:


model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(30))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100, return_sequences=True))
model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(Xtrain,Ytrain,epochs=50, verbose=1 )


# In[45]:


model.save("./regressor.hdf5")


# In[47]:


predict = model.predict(Xtest)


# In[48]:


predict = scalar.inverse_transform(predict)


# In[49]:


Ytesting = scalar.inverse_transform(Ytest)


# In[50]:


plt.figure(figsize=(20,9))
plt.plot(Ytesting , 'blue', linewidth=5)
plt.plot(predict,'r' , linewidth=4)
plt.legend(('Test','Predicted'))
plt.show()


# In[ ]:




