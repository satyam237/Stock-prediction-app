import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit  as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import math

st.title('Stock Trend Predictor')

user_input=st.text_input('Enter Stock Ticker', 'SBIN.NS')
start='2010-01-01'
end=dt.datetime.today()
df=yf.download(user_input,start=start,end=end)

#Describing Data
st.subheader('Data from 2010 - Present')
st.write(df.describe())
day_input=st.text_input('Enter number of days',10)
st.subheader('Data for last '+ day_input + ' days')
st.write(df.tail(int(day_input)))

#Visualization
st.subheader('Candlestick Graph')
candlestick = go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close']
                            )

fig3 = go.Figure(data=[candlestick])
fig3.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig3,use_container_width=True)


st.subheader('Closing Price vs Time Chart')
scatter = go.Scatter(
                    x=df.index,
                    y=df['Close']
                    )
fig = go.Figure(data=[scatter])
st.plotly_chart(fig,use_container_width=True)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
scatter1 = go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Closing Price'
                    )
scatter2 = go.Scatter(
                    x=ma100.index,
                    y=ma100[0:],
                    name='EMA100'
                    )
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(scatter1)
fig.add_trace(scatter2,secondary_y=True)
st.plotly_chart(fig,use_container_width=True)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200=df.Close.rolling(200).mean()
scatter3=go.Scatter(
                    x=ma200.index,
                    y=ma200[0:],
                    name='EMA200'
                    )
fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(scatter1)
fig4.add_trace(scatter2,secondary_y=True)
fig4.add_trace(scatter3,secondary_y=True)
st.plotly_chart(fig4,use_container_width=True)

#Splitting Data into Training and Testing
training_dt=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test_dt=pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
test_date=pd.DataFrame(df.index[int(len(df)*0.7):int(len(df))])
predict_date=pd.DataFrame(df.index[int(len(df)*0.7):int(len(df))])


scaler=MinMaxScaler(feature_range=(0,1))
training_dt_arr=scaler.fit_transform(training_dt)


#Load the Model
model=load_model('stock_model.h5')


#Testing Part
past_100_days=training_dt.tail(100)
final_df=pd.concat([past_100_days,test_dt])
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test=np.array(x_test)
# y_test=np.array(y_test)
y_predicted=model.predict(x_test)
# scaler=scaler.scale_

# scale_factor=1/scaler[0]
# y_predicted=y_predicted
# y_test=y_test*scale_factor
# y_test=np.reshape(-1,1)
# y_predicted=np.reshape(-1,1)
y_test=pd.DataFrame(y_test)
y_predicted=pd.DataFrame(y_predicted)
y_test=scaler.inverse_transform(y_test)
y_predicted=scaler.inverse_transform(y_predicted)



#Final Graph
st.subheader('Prediction vs Original')
# y_predicted=pd.DataFrame(y_predicted)
predict_date['Close']=y_predicted
predict_date.set_index('Date',inplace=True)

y_test=pd.DataFrame(y_test)
y_test.reset_index()
test_date['Close']=y_test
test_date.set_index('Date',inplace=True)

scatter4 = go.Scatter(
                    x=predict_date.index,
                    y=predict_date['Close'],
                    name='Predicted Price'
                    )
scatter5 = go.Scatter(
                    x=test_date.index,
                    y=test_date['Close'],
                    name='Original Price'
                    )
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(scatter4)
fig.add_trace(scatter5,secondary_y=True)
st.plotly_chart(fig,use_container_width=True)


col=st.columns(2)
col[0].markdown("Predicted Data")
col[0].dataframe(predict_date.tail(10))
col[1].markdown("Original Data")
col[1].dataframe(test_date.tail(10))

print("\nPredicted Values:\n")
print(predict_date.tail(10))
print("\nOriginal Values:\n")
print(test_date.tail(10))

mse=mean_squared_error(y_test,y_predicted)
rmse=math.sqrt(mse)
st.write("RMSE:",rmse)
print("\nRMSE:",rmse)
