from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox
from ctypes import windll
from PIL import Image, ImageTk
import yfinance as yf
import matplotlib.pyplot as plt
import mplcursors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, r2_score
import google.generativeai as genai
import os
import csv

windll.shcore.SetProcessDpiAwareness(1)

root = Tk()
root.title("Stock Market Predictor & Suggestor")
root.geometry("1500x900")
root.configure(bg="white")
root.resizable(False, False)

image_icon = PhotoImage(file="C:/Users/Priyanshu Madhup/Desktop/STOCK/s_logo2.png")
root.iconphoto(False, image_icon)

img = PhotoImage(file="C:/Users/Priyanshu Madhup/Desktop/STOCK/STOCK MARKET PREDICTOR AND SUGGESTOR (2).png")
img_loc = Label(root, image=img, bg='white').place(x=50, y=50)

simg = PhotoImage(file="C:/Users/Priyanshu Madhup/Desktop/STOCK/Untitled design (7).png")
s_img_loc = Label(root, image=simg, bg='white')
s_img_loc.place(x=150, y=300)

textfield = tk.Entry(root, justify='center', width=15, font=('poppins', 20, 'bold'), bg="white", border=0, fg="black")
textfield.place(x=180, y=320)
textfield.focus()

tk_label = Label(root, text="Enter Ticker Symbol", font=('poppins', 15), bg="white")
tk_label.place(x=180, y=220)

def func():
    tk_name = textfield.get()
    print(tk_name)
    stock = yf.Ticker(tk_name)
    ticker = tk_name

    ai_label = Label(root, text="AI's  SUGGESTION & ANALYSIS", font=('poppins', 15, 'bold'), bg="white")
    ai_label.place(x=230, y=225)
    
    s_img_loc.place(x=1050, y=100)
    textfield.place(x=1080, y=120)
    myimage_icon.place(x=1370, y=100)
    tk_label.place(x=1100, y=60)

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
        print(f"Predicted Closing Prices for the Next 30 Days:")
        for date, price in zip(future_dates, predicted_future_prices):
            print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
            writer.writerow([f"{date.strftime('%Y-%m-%d')}", f"{price:.2f}"])

    text_widget = tk.Text(root, height=13, width=70, bd=5)
    text_widget.place(x=100, y=270)

    text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
    
    genai.configure(api_key="AIzaSyDthMayeusJ9At_ofHC2_MEIyrTWJmEBEU")
    generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
    csv_file_path = file
    df = pd.read_csv(csv_file_path)
    #df = df.head()
    csv_content = df.to_string()

    model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

    response = model.generate_content([f"Give me some details about the company {tk_name}, some history and Summarize this data and if possible display the table, and also if possible give me some latest news about it " + csv_content])
    text_widget.insert(tk.END, response.text, "bold")
    print(response.text)

    #result_text = tk.Text(root, wrap='word', width=70, height=15)
    #result_text.place(x=100, y=500)

    def closing_graph():
        plt.figure(figsize=(10, 5))
        line1, = plt.plot(hist.index, closing_price, label='Close Price')
        plt.title(f'1 Year Historical Data for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        cursor1 = mplcursors.cursor(line1, hover=True)
        cursor1.connect("add", lambda sel: sel.annotation.set_text(f'{closing_price.iloc[min(max(int(sel.index), 0), len(closing_price) - 1)]:.2f}'))
        plt.show()

    def opening_graph():
        plt.figure(figsize=(10, 5))
        line2, = plt.plot(hist.index, opening_price, label='Open Price')
        plt.title(f'1 Year Historical Data for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.legend()
        plt.grid(True)
        cursor2 = mplcursors.cursor(line2, hover=True)
        cursor2.connect("add", lambda sel: sel.annotation.set_text(f'{opening_price.iloc[min(max(int(sel.index), 0), len(opening_price) - 1)]:.2f}'))
        plt.show()

    def predicted_graph():
        plt.figure(figsize=(10, 5))
        line3, = plt.plot(hist.index, closing_price, label='Historical Close Price', color='blue')
        line4, = plt.plot(future_dates, predicted_future_prices, label='Predicted Close Price (Next 30 days)', color='red', linestyle='dashed')
        plt.title(f'Historical and Predicted Closing Prices for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        cursor3 = mplcursors.cursor(line3, hover=True)
        cursor4 = mplcursors.cursor(line4, hover=True)
        cursor3.connect("add", lambda sel: sel.annotation.set_text(f'Date: {hist.index[min(max(int(sel.index), 0), len(hist.index) - 1)].strftime("%Y-%m-%d")}\nClose Price: {closing_price.iloc[min(max(int(sel.index), 0), len(closing_price) - 1)]:.2f}'))
        cursor4.connect("add", lambda sel: sel.annotation.set_text(f'Date: {future_dates[min(max(int(sel.index), 0), len(future_dates) - 1)].strftime("%Y-%m-%d")}\nPredicted Price: {predicted_future_prices[min(max(int(sel.index), 0), len(predicted_future_prices) - 1)]:.2f}'))
        plt.show()

    clp_bu = Button(text="CLOSING PRICE", cursor="hand2", bg="black", fg="white", command=closing_graph)
    clp_bu.place(x=180, y=620)
    clp_bu = Button(text="OPENING PRICE", cursor="hand2", bg="black", fg="white", command=opening_graph)
    clp_bu.place(x=380, y=620)
    clp_bu = Button(text="PREDICTED PRICE", cursor="hand2", bg="black", fg="white", command=predicted_graph)
    clp_bu.place(x=180, y=720)
    clp_bu = Button(text="EXIT", cursor="hand2", bg="red", fg="black", command=root.destroy)
    clp_bu.place(x=380, y=720)

s_icon = PhotoImage(file="C:/Users/Priyanshu Madhup/Desktop/STOCK/s_icon2.png")
myimage_icon = Button(image=s_icon, borderwidth=5, cursor="hand2", bg="white", command=func)
myimage_icon.place(x=460, y=300)

root.mainloop()