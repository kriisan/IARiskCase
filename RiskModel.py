from copy import copy
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import random
import time
import matplotlib.pyplot as plt
# assumption
# 1. returns for each day is independent of return from previous day
#       since there are days when US market is open and Canada isn't and vice versa to simply program will only use
#       dates where returns can be calculated for both markets. Future improvement would be use linear interpolation of
#       asset price over missing date then calculate return to fill gaps in data.

#variables
datefromat = '%Y-%m-%d'
startDateData = dt.date(2017,1,1).strftime(datefromat)      #start date of historical data
endDateData = dt.date(2021,12,31).strftime(datefromat)      #End date of historical data
forecastStartDate = dt.date(2022,3,31).strftime(datefromat)     #start date for forecase
iteration = 5000        #Number of iterations (trials)
numprd = 4              #Number of periods stich together
szprd = 62              #size of periods, since 250 buisness days in a year approx 250/4=62.5
identiferCCy = []

# Function to load portfolio position and return dataframe to be used in risk analysis
def load_portfolio():
    # identifier -> how product is referenced
    # identifierType -> example ticker/ISIN/CUSIP
    # productType -> equity/bond/swap
    # baseCurrency -> currency in which product is priced in
    # positionValueCAD -> value of position converted to CAD (typically would get position value in base currency and
    # use current FX rate to convert. CAD simplifies example to maintain a starting allocation of 50% for both products)
    # build global array that has ticker and ccy pair to be able to apply fx correctly
    global identiferCCy
    df = pd.read_csv('portfolio.csv')
    identiferCCy = df[['identifier','baseCurrency']].to_numpy()
    return df



# return numpy array of daily close price for currency pair
def fx_rate(pair, startDate, endDate):
    return yf.Ticker(pair).history(start= startDate, end= endDate)["Close"].to_numpy()

# build dataframe for each type of risk factor with daily returns
# input start and end of date for historical data set and dataframe with only equities, returns a dataframe with the
# daily returns for each equity
def equity_daily_return(equity, startDate, endDate):
    equityReturns = []
    # yahoo finance to get price on date T the end date needs to be date T+1
    endDate = (dt.datetime.strptime(endDate, datefromat) + dt.timedelta(days=1)).strftime(datefromat)
    for i in range(0, equity.shape[0]):
        # get historical price data
        # add error catching for tickers that don't work
        # get last quote of 2016?
        if equity[i:i+1]['baseCurrency'].values == 'CAD':
            historicalPrice = yf.Ticker(equity[i:i+1]['identifier'].values[0] + '.to').history(start= startDate, end= endDate)["Close"]
        elif equity[i:i+1]['baseCurrency'].values == 'USD':
            historicalPrice = yf.Ticker(equity[i:i+1]['identifier'].values[0]).history(start= startDate, end= endDate)["Close"]

        # calculate daily return current close price divided by previous day close price
        historicalPriceValue = historicalPrice.to_numpy()
        daily_return = [historicalPriceValue[i] / historicalPriceValue[i - 1] for i in range(1, len(historicalPriceValue))]
        # added temp filler 0 to first row of dataset to allow for date index to remain intact, added to beginning since
        # return formula uses close price so daily return on start date isn't calculated
        daily_return.insert(0, 0)
        temp = pd.DataFrame({equity[i:i+1]['identifier'].values[0]: daily_return}, index=historicalPrice.index).iloc[1: , :]
        # if first set of returns then set it as value of result else inner join with existing results
        if len(equityReturns) == 0:
            equityReturns = temp
        else:
            # Canada and US have different holidays so daily returns missing for certain, using inner join to ensure
            # dates used all have return values
            equityReturns = pd.concat([equityReturns, temp], axis = 1, join='inner')

    return equityReturns

# input start and end of date for historical data set, returns a dataframe with the daily returns for CAD/USD rate apart
# from return of day 1
def fx_daily_change(startDate, endDate):
    # get currency pairs for dates used base/quote
    eurcad = yf.Ticker('EURCAD=X').history(start= startDate, end= endDate)["Close"]
    eurusd = yf.Ticker('EURUSD=X').history(start= startDate, end= endDate)["Close"]
    # ensure fx rate available for all dates used
    tempFX = pd.concat([eurcad, eurusd], axis=1, join='inner')
    tempFX.columns = ['EURCAD', 'EURUSD']
    # calculate CAD/USD rate using ((EUR/CAD)^-1)*(EUR/USD)
    cadusd = np.multiply(np.power(tempFX['EURUSD'].values,-1),tempFX['EURCAD'].values)
    # calculate daily return current close price divided by previous day close price
    daily_return = [cadusd[i] / cadusd[i - 1] for i in range(1, len(cadusd))]
    # added temp filler 0 to first row of dataset to allow for date index to remain intact, added to beginning since
    # return formula uses close price so daily return on start date isn't calculated
    daily_return.insert(0, 0)
    #convert result into dataframe
    fxDailyChange = pd.DataFrame({'cadusd': daily_return}, index=tempFX.index).iloc[1: , :]
    return fxDailyChange

# funtion that will return initial price in data frame given portfolio input
def intial_price (port, date):
    equities = port.loc[portfolio['productType'] == "equity"]
    intialPrice = []
    # yahoo finance to get price on date T the end date needs to be date T+1
    dateP1 = (dt.datetime.strptime(date, datefromat) + dt.timedelta(days=1)).strftime(datefromat)
    for i in range(0, equities.shape[0]):
        if equities[i:i + 1]['baseCurrency'].values == 'CAD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0] + '.to').history(start=date,end=dateP1)["Close"]
        elif equities[i:i + 1]['baseCurrency'].values == 'USD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0]).history(start=date,end=dateP1)["Close"]
        temp = pd.DataFrame({equities[i:i + 1]['identifier'].values[0]: price.values})
        if len(intialPrice) == 0:
            intialPrice = temp
        else:
            intialPrice = pd.concat([intialPrice, temp], axis=1, join='inner')

    return intialPrice

def intial_fx (date):

    intailFX = []
    cadusdIntial = np.multiply(np.reciprocal(fx_rate('EURUSD=X', date, date)),
                               fx_rate('EURCAD=X', date, date))
    temp = pd.DataFrame({'cadusd': cadusdIntial})
    if len(intailFX) == 0:
        intialFX = temp
    else:
        intialFX = pd.concat([intailFX, temp], axis=1, join='inner')
    return intialFX

# funtion that will accept daily return and initial values of risk factors to return the forecasted results using the
# following method
# select m number of random sets of n returns and stitch returns together and return final values of factors
def forecast_risk_factor (intialValueEquity, dailyReturnEquity, intialValueFX, dailyReturnFX, m, n):
    forecast = np.zeros((m*n + 1, dailyReturnEquity.shape[1]))
    forecastFX = np.zeros((m*n + 1, dailyReturnFX.shape[1]))
    forecast[0] = intialValueEquity.values
    forecastFX[0] = intialValueFX.values
    for i in range(0,m):
        start = random.randint(0, dailyReturnEquity.shape[0] - n)
        offset = np.where(forecast == 0)[0][0]-1
        for j in range(start, start+n):
            forecast[j+1-start+offset] = forecast[j-start+offset]*dailyReturnEquity.values[j]
            forecastFX[j + 1 - start + offset] = forecastFX[j - start + offset] * dailyReturnFX.values[j]

    return forecast.transpose(), forecastFX.transpose()



portfolio = load_portfolio()
equities = portfolio.loc[portfolio['productType'] == "equity"]
equityReturns = equity_daily_return(equities,startDateData,endDateData)
fxDailyChange = fx_daily_change(startDateData, endDateData)
# Add check for if factors not empty then concat

# some FX data points missing from yahoo finance, for example 2017-11-16, using inner join to ensure data avaliable for
# all dates used in forecast
riskFactorInnerJoin = pd.concat([equityReturns, fxDailyChange], axis=1, join='inner')
equityReturns = riskFactorInnerJoin[equityReturns.columns]
fxDailyChange = riskFactorInnerJoin[fxDailyChange.columns]


#Intial price of risk factors
intialPrice = intial_price(portfolio, forecastStartDate)
intialFX = intial_fx(forecastStartDate)


intialPos = intialPrice[portfolio['identifier'].values]

for i in intialPos.columns:
    # attempt to calculate position only if header name exists in identifier ccy pair table, this ensures to not do
    # calculation on fx rate
    try:
        if identiferCCy[np.argwhere(identiferCCy == i)[0][0]][1] == 'USD':
            intialPos.at[0,i] *= intialFX.at[0,'cadusd']
        # get index of ticker in portfolio
        k = portfolio[portfolio['identifier'] == i].index.values[0]
        intialPos.at[0,i] = portfolio.at[k,'positionValueCAD']/intialPos.at[0,i]

    except:
        pass


# add checks to ensure riskfactors exist for each initial price

# forcast results
numAssets = equityReturns.shape[1]
forecasts = np.zeros((iteration*numAssets,numprd*szprd+1))
forecastsFX = np.zeros((iteration*fxDailyChange.shape[1],numprd*szprd+1))
for i in range(0, iteration):
    forecasts[i*numAssets:i*numAssets+2], forecastsFX[i:i+1] = forecast_risk_factor(intialPrice, equityReturns, intialFX, fxDailyChange, numprd, szprd)

portfolioValue = np.zeros((iteration,numprd*szprd+1))
for i in range(0,iteration):
    forecastsTemp = forecasts[i*numAssets:i*numAssets+2]
    # apply fx to USD stocks using the shape of portfolio to determine which columns refer to US stocks
    forecastsTemp[(portfolio['baseCurrency'].values == 'USD')] = forecastsTemp[(portfolio['baseCurrency'].values == 'USD')]*forecastsFX[i]
    tempIntialVal = (intialPos.values).flatten()
    for j in range(0,numprd*szprd+1):
        for k in range(0, numAssets):
            portfolioValue[i][j] += forecastsTemp[k][j]*tempIntialVal[k]

portfolioReturns = np.zeros((iteration,numprd*szprd))
portfolioTotalReturn = np.zeros((iteration,1))
for i in range(0,iteration):
    portfolioTotalReturn[i] = portfolioValue[i][-1]/portfolioValue[i][0] - 1
    for j in range(0, numprd * szprd):
        portfolioReturns[i][j] = portfolioValue[i][j+1]/portfolioValue[i][j] - 1

print('Average expected return: ', np.average(portfolioTotalReturn))
#
# Calculate Expected 1% CVaR (Expected shortfall)
# np.sort(portfolioTotalReturn)
print('1% CVar: ', np.average(np.percentile(portfolioTotalReturn,1)))

np.sort(portfolioTotalReturn)
print('1% CVar: ', np.average(np.percentile(portfolioTotalReturn,1)))


