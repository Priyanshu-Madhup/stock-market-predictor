# # import streamlit as st
# # import yfinance as yf
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import numpy as np
# # from datetime import timedelta
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_absolute_error, r2_score

# # # Streamlit app title
# # st.title("Stock Market Predictor & Suggestor")

# # # Sidebar for user input (ticker symbol)
# # st.sidebar.header("Input Ticker Symbol")
# # ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOG)", "AAPL")

# # # Function to get historical data and perform predictions
# # def predict_stock(ticker_symbol):
# #     stock = yf.Ticker(ticker_symbol)
    
# #     # Get historical data for the past 1 year
# #     hist = stock.history(period="1y")

# #     if hist.empty:
# #         st.error("No data found for the given ticker symbol. Please try another symbol.")
# #         return None
    
# #     st.write(f"### Historical Data for {ticker_symbol} (Last 1 year)")
# #     st.write(hist.tail())  # Show the last few rows of the historical data
    
# #     # Preparing the dataset for predictions
# #     closing_price = hist['Close']
# #     hist.index = hist.index.tz_localize(None)  # Ensure the index is timezone-naive
# #     hist['Date'] = (pd.to_datetime(hist.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# #     hist['DayOfWeek'] = hist.index.dayofweek
# #     hist['Month'] = hist.index.month

# #     X = hist[['Date', 'DayOfWeek', 'Month']]  # Features
# #     y = hist['Close']  # Target variable

# #     # Split the data into training and testing sets
# #     X_train_cp, X_test_cp, y_train_cp, y_test_cp = train_test_split(X, y, random_state=42)

# #     # Train the model
# #     rf = RandomForestRegressor()
# #     rf.fit(X, y)

# #     # Generate future dates for the next 30 days
# #     last_date = hist.index[-1]
# #     future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
# #     future_timestamps = [(pd.Timestamp(date) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for date in future_dates]

# #     # Prepare feature set for future predictions
# #     future_data = pd.DataFrame({
# #         'Date': future_timestamps,
# #         'DayOfWeek': [date.weekday() for date in future_dates],
# #         'Month': [date.month for date in future_dates]
# #     })

# #     # Predict future closing prices
# #     predicted_future_prices = rf.predict(future_data)

# #     return hist, closing_price, future_dates, predicted_future_prices

# # # Call the prediction function
# # if ticker_symbol:
# #     hist, closing_price, future_dates, predicted_future_prices = predict_stock(ticker_symbol)

# #     if hist is not None:
# #         # Plot historical closing prices
# #         st.write("### Historical Closing Price")
# #         fig, ax = plt.subplots()
# #         ax.plot(hist.index, closing_price, label='Historical Close Price', color='blue')
# #         ax.set_xlabel("Date")
# #         ax.set_ylabel("Close Price")
# #         ax.set_title(f"Historical Close Price for {ticker_symbol}")
# #         st.pyplot(fig)

# #         # Plot predicted future closing prices
# #         st.write("### Predicted Closing Prices for the Next 30 Days")
# #         fig, ax = plt.subplots()
# #         ax.plot(hist.index, closing_price, label='Historical Close Price', color='blue')
# #         ax.plot(future_dates, predicted_future_prices, label='Predicted Close Price', color='red', linestyle='dashed')
# #         ax.set_xlabel("Date")
# #         ax.set_ylabel("Close Price")
# #         ax.set_title(f"Predicted Closing Prices for {ticker_symbol}")
# #         st.pyplot(fig)

# #         # Display predicted prices in a table
# #         st.write("### Predicted Prices (Next 30 Days)")
# #         predicted_df = pd.DataFrame({
# #             'Date': future_dates,
# #             'Predicted Closing Price': predicted_future_prices
# #         })
# #         st.write(predicted_df)

# # # To run the app, use the following command in your terminal:
# # # streamlit run stream1.py
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from datetime import timedelta
# import google.generativeai as genai
# import csv

# # Set up the Streamlit app layout
# st.title("Stock Market Predictor & Suggestor")
# #st.image("C:/Users/Priyanshu Madhup/Desktop/STOCK/STOCK MARKET PREDICTOR AND SUGGESTOR (2).png")
# #st.image("C:/Users/Priyanshu Madhup/Desktop/STOCK/Untitled design (7).png")

# # Create input fields and buttons
# tk_name = st.text_input("Enter Ticker Symbol", "")
# if st.button("Predict"):
#     if tk_name:
#         st.write(f"Ticker Symbol: {tk_name}")
#         stock = yf.Ticker(tk_name)
#         ticker = tk_name

#         st.subheader("AI's SUGGESTION & ANALYSIS")

#         # Get historical market data
#         hist = stock.history(period="1y")

#         # Accessing specific columns from the hist dataframe
#         closing_price = hist['Close']
#         opening_price = hist['Open']

#         hist.index = hist.index.tz_localize(None)  # Ensure the index is timezone-naive
#         hist['Date'] = (pd.to_datetime(hist.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#         hist['DayOfWeek'] = hist.index.dayofweek
#         hist['Month'] = hist.index.month

#         X = hist[['Date', 'DayOfWeek', 'Month']]  # Features
#         y = hist['Close']   # Target variable

#         # Split the data into training and testing sets
#         X_train_cp, X_test_cp, y_train_cp, y_test_cp = train_test_split(X, y, random_state=42)

#         rf = RandomForestRegressor()
#         rf.fit(X, y)

#         # Generate future dates for the next 30 days
#         last_date = hist.index[-1]
#         future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
#         future_timestamps = [(pd.Timestamp(date) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for date in future_dates]

#         # Prepare feature set for future predictions
#         future_data = pd.DataFrame({
#             'Date': future_timestamps,
#             'DayOfWeek': [date.weekday() for date in future_dates],
#             'Month': [date.month for date in future_dates]
#         })

#         file = "stock_pred.csv"
#         with open(file, "w", newline='') as f:
#             writer = csv.writer(f)
#             # Write the header for the CSV file
#             writer.writerow(["DATE", "PREDICTED CLOSING PRICE"])

#             # Predict future closing prices
#             predicted_future_prices = rf.predict(future_data)
#             predicted_future_prices_a = np.array(predicted_future_prices)
#             # Print predicted values
#             st.write("Predicted Closing Prices for the Next 30 Days:")
#             for date, price in zip(future_dates, predicted_future_prices):
#                 st.write(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
#                 writer.writerow([f"{date.strftime('%Y-%m-%d')}", f"{price:.2f}"])

#         genai.configure(api_key="AIzaSyDthMayeusJ9At_ofHC2_MEIyrTWJmEBEU")
#         generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
#         csv_file_path = file
#         df = pd.read_csv(csv_file_path)
#         csv_content = df.to_string()

#         model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

#         response = model.generate_content([f"Give me some details about the company {tk_name}, some history and Summarize this data and if possible display the table, and also if possible give me some latest news about it " + csv_content])
#         st.write(response.text)

#         # Plot graphs
#         def plot_closing_graph():
#             plt.figure(figsize=(10, 5))
#             plt.plot(hist.index, closing_price, label='Close Price')
#             plt.title(f'1 Year Historical Data for {ticker}')
#             plt.xlabel('Date')
#             plt.ylabel('Close Price')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)

#         def plot_opening_graph():
#             plt.figure(figsize=(10, 5))
#             plt.plot(hist.index, opening_price, label='Open Price')
#             plt.title(f'1 Year Historical Data for {ticker}')
#             plt.xlabel('Date')
#             plt.ylabel('Open Price')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)

#         def plot_predicted_graph():
#             plt.figure(figsize=(10, 5))
#             plt.plot(hist.index, closing_price, label='Historical Close Price', color='blue')
#             plt.plot(future_dates, predicted_future_prices, label='Predicted Close Price (Next 30 days)', color='red', linestyle='dashed')
#             plt.title(f'Historical and Predicted Closing Prices for {ticker}')
#             plt.xlabel('Date')
#             plt.ylabel('Close Price')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)

#         if st.button("Show Closing Price Graph"):
#             plot_closing_graph()
#         if st.button("Show Opening Price Graph"):
#             plot_opening_graph()
#         if st.button("Show Predicted Price Graph"):
#             plot_predicted_graph()
#     else:
#         st.error("Please enter a valid ticker symbol.")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import google.generativeai as genai
import csv

# Set up the Streamlit app layout
st.title("Stock Market Predictor & Suggestor")
#st.image("C:/Users/Priyanshu Madhup/Desktop/STOCK/STOCK MARKET PREDICTOR AND SUGGESTOR (2).png")
#st.image("C:/Users/Priyanshu Madhup/Desktop/STOCK/Untitled design (7).png")

# Initialize session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = ''
if 'show_closing' not in st.session_state:
    st.session_state.show_closing = False
if 'show_opening' not in st.session_state:
    st.session_state.show_opening = False
if 'show_predicted' not in st.session_state:
    st.session_state.show_predicted = False
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Create input fields and buttons
tk_name = st.text_input("Enter Ticker Symbol", st.session_state.ticker)
if st.button("Predict"):
    if tk_name:
        st.session_state.ticker = tk_name
        st.session_state.show_closing = False
        st.session_state.show_opening = False
        st.session_state.show_predicted = False
        st.session_state.predicted = True

        stock = yf.Ticker(tk_name)
        ticker = tk_name

        st.write(f"Ticker Symbol: {tk_name}")
        st.subheader("AI's SUGGESTION & ANALYSIS")

        # Get historical market data
        hist = stock.history(period="1y")

        # Accessing specific columns from the hist dataframe
        closing_price = hist['Close']
        opening_price = hist['Open']

        hist.index = hist.index.tz_localize(None)  # Ensure the index is timezone-naive
        hist['Date'] = (pd.to_datetime(hist.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        hist['DayOfWeek'] = hist.index.dayofweek
        hist['Month'] = hist.index.month

        X = hist[['Date', 'DayOfWeek', 'Month']]  # Features
        y = hist['Close']   # Target variable

        # Split the data into training and testing sets
        X_train_cp, X_test_cp, y_train_cp, y_test_cp = train_test_split(X, y, random_state=42)

        rf = RandomForestRegressor()
        rf.fit(X, y)

        # Generate future dates for the next 30 days
        last_date = hist.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        future_timestamps = [(pd.Timestamp(date) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for date in future_dates]

        # Prepare feature set for future predictions
        future_data = pd.DataFrame({
            'Date': future_timestamps,
            'DayOfWeek': [date.weekday() for date in future_dates],
            'Month': [date.month for date in future_dates]
        })

        file = "stock_pred.csv"
        with open(file, "w", newline='') as f:
            writer = csv.writer(f)
            # Write the header for the CSV file
            writer.writerow(["DATE", "PREDICTED CLOSING PRICE"])

            # Predict future closing prices
            predicted_future_prices = rf.predict(future_data)
            predicted_future_prices_a = np.array(predicted_future_prices)
            # Print predicted values
            st.write("Predicted Closing Prices for the Next 30 Days:")
            for date, price in zip(future_dates, predicted_future_prices):
                st.write(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
                writer.writerow([f"{date.strftime('%Y-%m-%d')}", f"{price:.2f}"])

        genai.configure(api_key="AIzaSyDthMayeusJ9At_ofHC2_MEIyrTWJmEBEU")
        generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
        csv_file_path = file
        df = pd.read_csv(csv_file_path)
        csv_content = df.to_string()

        model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

        response = model.generate_content([f"Give me some details about the company {tk_name}, some history and Summarize this data and if possible display the table, and also if possible give me some latest news about it " + csv_content])
        st.write(response.text)

        # Store data in session state
        st.session_state.hist = hist
        st.session_state.closing_price = closing_price
        st.session_state.opening_price = opening_price
        st.session_state.future_dates = future_dates
        st.session_state.predicted_future_prices = predicted_future_prices

# Plot graphs using Plotly
def plot_closing_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist.index, y=st.session_state.closing_price, mode='lines', name='Close Price'))
    fig.update_layout(title=f'1 Year Historical Data for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

def plot_opening_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist.index, y=st.session_state.opening_price, mode='lines', name='Open Price'))
    fig.update_layout(title=f'1 Year Historical Data for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig)

def plot_predicted_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist.index, y=st.session_state.closing_price, mode='lines', name='Historical Close Price'))
    fig.add_trace(go.Scatter(x=st.session_state.future_dates, y=st.session_state.predicted_future_prices, mode='lines', name='Predicted Close Price (Next 30 days)', line=dict(dash='dash')))
    fig.update_layout(title=f'Historical and Predicted Closing Prices for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

# Display graph buttons only after prediction
if st.session_state.predicted:
    if st.button("Show Closing Price Graph"):
        st.session_state.show_closing = True
        st.session_state.show_opening = False
        st.session_state.show_predicted = False

    if st.button("Show Opening Price Graph"):
        st.session_state.show_closing = False
        st.session_state.show_opening = True
        st.session_state.show_predicted = False

    if st.button("Show Predicted Price Graph"):
        st.session_state.show_closing = False
        st.session_state.show_opening = False
        st.session_state.show_predicted = True

    # Display the selected graph
    if st.session_state.show_closing:
        plot_closing_graph()
    elif st.session_state.show_opening:
        plot_opening_graph()
    elif st.session_state.show_predicted:
        plot_predicted_graph()