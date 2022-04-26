import yfinance as yf
import pandas as pd
import numpy as np

# assumption
# 1. returns for each day is independent of return from previous day
#       since there are days when US market is open and Canada isn't and vice versa to simply program will only use
#       dates where returns can be calculated for both markets. Future improvement would be use linear interpolation of
#       asset price over missing date then calculate return to fill gaps in data.

#variables
startDateData = '2017-01-01'
endDateData = '2017-12-31'
forecastStartDate = '2022-03-31'

# Function to load portfolio position and return dataframe to be used in risk analysis
def load_portfolio():
    # identifier -> how product is referenced
    # identifierType -> example ticker/ISIN/CUSIP
    # productType -> equity/bond/swap
    # baseCurrency -> currency in which product is priced in
    # positionValueCAD -> value of position converted to CAD (typically would get position value in base currency and
    # use current FX rate to convert. CAD simplifies example to maintain a starting allocation of 50% for both products)
    return pd.read_csv('portfolio.csv')


# build dataframe for each type of risk factor with daily returns
# function to build array for equity risk factor daily return where each column is one equity following order of
def equity_daily_return(equity, startDate, endDate):
    equityReturns = []
    for i in range(0, equity.shape[0]):
        # get historical price data
        # add error catching for tickers that don't work
        # get last quote of 2016?
        if equity[i:i+1]['baseCurrency'].values == 'CAD':
            historicalPrice = yf.Ticker(equity[i:i+1]['identifier'].values[0] + '.to').history(start= startDate, end= endDate)["Close"]
        elif equity[i:i+1]['baseCurrency'].values == 'USD':
            historicalPrice = yf.Ticker(equity[i:i+1]['identifier'].values[0]).history(start= startDate, end= endDate)["Close"]

        # calculate daily return new price divided by previous day price
        historicalPriceValue = historicalPrice.to_numpy()
        daily_return = [historicalPriceValue[i] / historicalPriceValue[i - 1] for i in range(1, len(historicalPriceValue))]
        # added filler 0 to first row of dataset to allow for date index to remain intact, added to beginning since
        # return formula uses close price so daily return on start date isn't calculated
        daily_return.insert(0, 0)
        temp = pd.DataFrame({equity[i:i+1]['identifier'].values[0]: daily_return}, index=historicalPrice.index)
        if len(equityReturns) == 0:
            equityReturns = temp
        else:
            equityReturns = pd.concat([equityReturns, temp], axis = 1)

    return equityReturns

def fx_daily_change(startDate, endDate):
    eurcad = yf.Ticker('EURCAD=X').history(start= startDate, end= endDate)["Close"].to_numpy()
    eurusd = yf.Ticker('EURUSD=X').history(start= startDate, end= endDate)["Close"].to_numpy()
    cadusd = np.multiply(np.reciprocal(eurcad),eurusd)
    daily_return = [cadusd[i] / cadusd[i - 1] for i in range(1, len(cadusd))]
    daily_return.insert(0, 0)
    fxDailyChange = pd.DataFrame({'cadusd': daily_return}, index=yf.Ticker('EURCAD=X').history(start= startDate, end= endDate).index)
    return fxDailyChange

# Generate data for 5000 execution of portfolio forecasting
# select 4 random series of 65 returns and apply to current price to determine future price

portfolio = load_portfolio()
#print(portfolio)
equityReturns = equity_daily_return(portfolio.loc[portfolio['productType'] == "equity"],startDateData,endDateData)
fxDailyChange = fx_daily_change(startDateData, endDateData)

print(equityReturns)
print(fxDailyChange)

riskFactorReturn = pd.concat([equityReturns, fxDailyChange], axis=1, join='inner');
print(riskFactorReturn)