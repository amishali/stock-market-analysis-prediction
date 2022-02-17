import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns 
from datetime import date
from pandas import Series, DataFrame
import pandas_datareader as pdr
import streamlit as st
import keras
import tensorflow as tf
from keras import models
from keras.models import load_model
from plotly import graph_objs as go


start  = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title('Stock Market Analysis And Prediction')

user_input = st.text_input('Enter the stock ticker of the stock you want to predict')

st.subheader('Raw data')
stock = pdr.DataReader('AAPL' , 'yahoo' , start , today )
st.write(stock.head())
st.write(stock.tail())


# show the description of data
st.subheader('Statistical  description of the Datasets:-')
description=stock.describe()
st.write(description)
# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
st.line_chart(stock[["Open","Close"]])
# display plot of volume column in datasets
st.subheader('Graph of  total volume of stock being traded each day over the past year:-')
st.line_chart(stock['Volume'])


#daily return analysis 
st.subheader('Daily return analysis' )
#we will use the pct_change to find the percentage change for each day 
stock['Daily Return'] = stock['Close'].pct_change()
#lets visulaize the daily return 
st.line_chart(stock['Daily Return'])

#lets now calaculate the moving Average 
#Moving averages are usually calculated to identify the trend direction
#A 10-day moving average would average out the closing prices for the first 10 days as the first data point.
# The next data point would drop the earliest price,
# add the price on day 11 and take the average.
#lets calcuate the moving Average of the first 50 , 100 , 200 day 
# Pandas has a built-in rolling mean calculator

# Let's go ahead and plot out several moving averages
st.subheader('Moving Average  analysis' )
MA_day = [50,100,200]

for ma in MA_day:
    column_name = 'MA for %s days' %(str(ma))
    stock[column_name] = stock['Close'].rolling(ma).mean()

st.line_chart(stock[['Close','MA for 50 days','MA for 100 days','MA for 200 days']])


#splitting our dataset into training and testing 
#we will predicting on our close column 
data_training = pd.DataFrame(stock['Close'][0:int(len(stock)*0.70)])#starting at 0 index i want to go till 70 % of data for training 
#
data_testing = pd.DataFrame(stock['Close'][0:int(len(stock)*0.30)]) #testing 30% of the dataset 

#lets scale our dataset 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
 #convert our training data to array 
data_training_scaled = scaler.fit_transform(data_training)

#split our training data into xtrain and y train 
#Creating data stucture with 100 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 100 days before current day to predict 1 next output
x_train=[] #xtrain is our feature class  100 days is xtrain 
y_train=[] #ytrain is our predicted class .. the value that is suppose to be predicted  101 the 1 data is the ytrain 
for i in range(100,data_training_scaled.shape[0]):
            x_train.append(data_training_scaled[i-100:i])
            y_train.append(data_training_scaled[i,0]) 
#lets convert the xtrain and y train into numpy array  so as to provide this data to our lstm 
#lets convert the list to array 
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

#load our saved model
model = tf.keras.models.load_model("C:/Users\hp/Documents/Stock market project/keras_model.h5")
#model= load_model('keras_model.h5')

#testing part 
#lets append data training and data testing 
past_100_days = data_training.tail(100)
final_stock= past_100_days.append(data_testing, ignore_index=True)
 #scale our data testing 
final_stock_scaled = scaler.fit_transform(final_stock)

#define xtest and ytest from data testing 
x_test=[]
y_test=[]
for i in range(100,final_stock_scaled.shape[0]):
            x_test.append(final_stock_scaled[i-100:i])
            y_test.append(final_stock_scaled[i,0]) 
#lets convert the list to array 
x_test = np.array(x_test) 
y_test = np.array(y_test)   

#make prediction 
prediction = model.predict(x_test)

#scale up our value 
scaler = scaler.scale_  
scaler_factor = 1/scaler[0]
prediction = prediction *scaler_factor
y_test= y_test*scaler_factor

#lets plot the prediction
st.subheader('PREDICTION VS ORIGINAL PRICE ')
fig2 =plt.figure(figsize=(10,4),dpi=65)
plt.plot(y_test, 'b' , label='original Price')  
plt.plot(prediction , 'r' , label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
#st.pyplot(fig2)
#st.line_chart(fig2)
st.plotly_chart(fig2)





