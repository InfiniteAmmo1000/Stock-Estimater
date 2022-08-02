import streamlit
import datetime
import yfinance
import fbprophet
import fbprophet.plot
import plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

streamlit.title('Stock Forecast App')

stocks = ("GOOG", "AAPL", "MSFT", "TSLA", "AMZN")
selected_stock = streamlit.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yfinance.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = streamlit.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

streamlit.subheader('Raw data')
streamlit.write(data.tail())

def plot_raw_data():
	fig = plotly.Figure()
	fig.add_trace(plotly.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(plotly.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	streamlit.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = fbprophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

streamlit.subheader('Forecast data')
streamlit.write(forecast.tail())
    
streamlit.write(f'Forecast plot for {n_years} years')
fig1 = fbprophet.plot(m, forecast)
streamlit.plotly_chart(fig1)

streamlit.write("Forecast components")
fig2 = m.plot_components(forecast)
streamlit.write(fig2)
