# %% this code was made following the tutorial of Courses/Finalcial-Python

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import MetaTrader5 as mt
import pandas_datareader as pdr
import yfinance as yf
import math as math
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose 



st.title("Finance Model")

st.write("""

    ## <- Open the sidebar to search the stock symbol

""")

# %% --------------------- Definindo os inputs de tempo e de stock  ---------------------
period_values_array = ["1d", "5d", "1mo", "3mo", 
                "6mo", "1y", "2y", "5y",
                "10y", "ytd", "max"]

symbol = st.sidebar.text_input("Search your stock symbol")
#period_value = st.sidebar.radio(options=period_values_array, label = "Period of stock symbol")
start_date = st.sidebar.text_input("Write the START date: year-month-day")
end_date = st.sidebar.text_input("Write the END date: year-month-day")

#                    --------------------- ---------------------


# %% --------------------- Função ativada quando o stock symbol existir  ---------------------
def model_activate():
    now = date.today()
    start_date = "2019-05-01"
    stock = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    ticker = yf.Tickers(symbol)
    if stock.empty:
        st.write("""
            # Couldn't find the stock symbol! Try again
        """)
        return
    concat_object = f'ticker.tickers.{symbol}.info["shortName"]'
    st.write("""
        ## Name of The Company is: """ + eval(concat_object))

    frequency = math.floor(len(stock['Adj Close'])/2)
 

    #pd.Index(sm.tsa.datetools.dates_from_range(f'2019Q1', f'2021Q3'))
    #if not mt.initialize():
    #    mt.initialize() 
    #df = sm.datasets.macrodata.load_pandas().data

    #st.write(df)

    st.write("""
        ### Dataset description
    """)

    #st.write(sm.datasets.macrodata.NOTE)

    # st.write("""
    #     ### Transform the Year column into a timeseries index with statsmodels tool
    # """)

    # index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
    # st.write(index)

    #df.index = index
    #st.write(df.head())

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    close = 'Adj Close'

    gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(stock[close])
    stock['trend'] = gdp_trend
    stock['6-month-SMA'] = stock[close].rolling(window = 6).mean()
    stock['12-month-SMA'] = stock[close].rolling(window = 12).mean()


    ax.plot(stock[close], color="red", label = close)
    ax.plot(stock['trend'], color="brown", label = 'trend')
    ax.plot(stock['6-month-SMA'], color="green", label = '6-month-SMA')
    ax.plot(stock['12-month-SMA'], color="blue", label = '12-month-SMA')
    ax.legend()
    ax.margins(x=0.001, y=0.001)
    st.pyplot(fig)


    st.write("""
        # Implementing the Exponential Weighted Moving Average
        """)

    stock['EWMA-12'] = stock[close].ewm(span=12).mean()
    ax2.plot(stock['EWMA-12'], color='green', label = 'EWMA-12')
    ax2.plot(stock[close], color='blue', label = close)
    ax2.legend()
    st.pyplot(fig2)


    st.write("""
        # ETS Model (Error-Trend-Seasonality)
    """)

    ets = seasonal_decompose(stock[close], model= 'additive', period=frequency, extrapolate_trend='freq')

    fig3, ax3 = plt.subplots()
    ax3.plot(ets.observed, color="blue", label="observed")
    ax3.plot(ets.trend, color="purple", label = 'trend')
    ax3.plot(ets.seasonal, color="orange", label= "seasonality")
    ax3.legend()
    st.pyplot(fig3)



    st.write("""
        #### Take note that to the seasoality to work propely we must do the following arithmetic equation:
        #### fixed_seasonality = number_of_collumns/2
    """)

    st.write("""
        # Arima Model (Auto Regressive Integrated Moving Averages)
    """)


# %%     
if symbol != "" and start_date != "" and end_date != "":
    model_activate()
else:
    st.write("""
        ## Fix your input!
    """)

    