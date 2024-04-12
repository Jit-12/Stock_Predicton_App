import os
import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
import ta
import requests
from dotenv import load_dotenv

def Select_stock():
    stock_names = ["AAPL", "GOOGL", "NVDA", "SPOT", "TSLA"]
    options = stock_names + ["Other"]

    stock_name = st.sidebar.selectbox("Select or type a stock", options)

    if stock_name == "Other":
        custom_name = st.sidebar.text_input("Enter the stock name: ")
        if custom_name:
            stock_name = custom_name.upper()

    return stock_name

def fetch_stock_data(stock_name, start_date, end_date):
    try:
        stock_data = yf.download(stock_name, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def get_news(symbol):
    try:
        load_dotenv()
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news_data = response.json()
        articles = news_data['articles']
        return articles
    except requests.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

def run_stock_prediction():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2620/2620564.png")
    st.sidebar.header("Stock Prediction App")
    start_date = st.sidebar.date_input("Start Date", datetime.now(), max_value=datetime.now())
    end_date = st.sidebar.date_input("End Date", datetime.now(), max_value=datetime.now())
    
    stock_name = Select_stock()

    graph_checkbox = st.sidebar.checkbox("Stock Price Graph")
    short_rolling_checkbox = st.sidebar.checkbox("Short Rolling Mean Graph")
    long_rolling_checkbox = st.sidebar.checkbox("Long Rolling Mean Graph")
    sma_checkbox = st.sidebar.checkbox("Simple Moving Average (SMA) Graph")
    rsi_checkbox = st.sidebar.checkbox("Relative Strength Index (RSI) Graph")
    macd_checkbox = st.sidebar.checkbox("Moving Average Convergence Divergence (MACD) Graph")
    bollinger_checkbox = st.sidebar.checkbox("Bollinger Bands Graph")
    prediction_years = st.sidebar.slider("Years of Prediction", 0, 3, 1)

    if st.sidebar.button("Run"):
        stock_data = fetch_stock_data(stock_name, start_date, end_date)
        if stock_data is not None:
            weekdays = pd.date_range(start=start_date, end=end_date)
            clean_data = stock_data['Adj Close'].reindex(weekdays)
            adj_close = clean_data.fillna(method='ffill')

            if graph_checkbox:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close.values, mode='lines', name='Adj Close',
                                        line=dict(color='royalblue')))
                fig.update_layout(title="Stock Price Graph", xaxis_title="Date", yaxis_title="Adj Close Price ($)")
                st.plotly_chart(fig)

            if short_rolling_checkbox:
                short_rolling_mean = adj_close.rolling(window=50).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=short_rolling_mean.index, y=short_rolling_mean.values, mode='lines', name='50-day Rolling Mean',
                                        line=dict(color='green')))
                fig.update_layout(title="Short Rolling Mean Graph", xaxis_title="Date", yaxis_title="50-day Rolling Mean")
                st.plotly_chart(fig)

            if long_rolling_checkbox:
                long_rolling_mean = adj_close.rolling(window=200).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=long_rolling_mean.index, y=long_rolling_mean.values, mode='lines', name='200-day Rolling Mean',
                                        line=dict(color='orange')))
                fig.update_layout(title="Long Rolling Mean Graph", xaxis_title="Date", yaxis_title="200-day Rolling Mean")
                st.plotly_chart(fig)

            if sma_checkbox:
                sma = adj_close.rolling(window=20).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sma.index, y=sma.values, mode='lines', name='SMA',
                                        line=dict(color='purple')))
                fig.update_layout(title="Simple Moving Average (SMA) Graph", xaxis_title="Date", yaxis_title="SMA Price ($)")
                st.plotly_chart(fig)

            if rsi_checkbox:
                rsi = ta.momentum.rsi(adj_close)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, mode='lines', name='RSI',
                                        line=dict(color='red')))
                fig.update_layout(title="Relative Strength Index (RSI) Graph", xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig)

            if macd_checkbox:
                macd = ta.trend.macd_diff(adj_close)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=macd.index, y=macd.values, mode='lines', name='MACD',
                                        line=dict(color='blue')))
                fig.update_layout(title="Moving Average Convergence Divergence (MACD) Graph", xaxis_title="Date", yaxis_title="MACD")
                st.plotly_chart(fig)

            if bollinger_checkbox:
                bollinger = ta.volatility.bollinger_mavg(adj_close)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bollinger.index, y=bollinger.values, mode='lines', name='Bollinger Bands',
                                        line=dict(color='green')))
                fig.update_layout(title="Bollinger Bands Graph", xaxis_title="Date", yaxis_title="Bollinger Bands")
                st.plotly_chart(fig)

            
            if prediction_years:
                pdata = yf.download(stock_name, start=start_date, end=end_date)
                pdata.reset_index(inplace=True)
                df_train = pdata[['Date', 'Close']]
                df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=prediction_years*365)
                forecast = m.predict(future)
                st.markdown("<h1 style='font-size: 20px;'>Prediction Graph</h1>", unsafe_allow_html=True)
                trace_pred = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price',
                                        line=dict(color='red'))
                trace_lower = go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound',
                                        line=dict(color='rgba(255, 0, 0, 0.3)'))
                trace_upper = go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound',
                                        line=dict(color='rgba(255, 0, 0, 0.3)'))

                trace_fill = go.Scatter(x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                                        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                                        fill='toself', fillcolor='rgba(255, 0, 0, 0.1)',
                                        line=dict(color='rgba(255, 0, 0, 0)'))

                layout = go.Layout(xaxis_title="Date", yaxis_title="Predicted Price",
                                hovermode='closest')

                fig = go.Figure(data=[trace_pred, trace_lower, trace_upper, trace_fill], layout=layout)

                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=0, r=0, t=40, b=0),
                                plot_bgcolor='rgba(0,0,0,0)')

                fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])),
                    rangeslider=dict(visible=True), type="date"))

                st.plotly_chart(fig)

        else:
            st.warning("No stock data available. Please try again with a different stock name or date range.")


def run_news_feed():
    st.title("News Feed")
    stock_name = Select_stock()
    articles = get_news(stock_name)
    for article in articles:
        st.write(f"**{article['title']}**")
        st.write(article['description'])
        st.write(f"Published on: {article['publishedAt']}")
        st.write(f"Source: {article['source']['name']}")
        st.write("---")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Stock Prediction", "News Feed"])
    if app_mode == "Stock Prediction":
        run_stock_prediction()
    elif app_mode == "News Feed":
        run_news_feed()


if __name__ == "__main__":
    main()
