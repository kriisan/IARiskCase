import yfinance as yf
import pandas as pd
import numpy as np

# Build Portfolio position

portfolio = pd.read_csv('portfolio.csv')

print(portfolio)
print(portfolio.dtypes)

