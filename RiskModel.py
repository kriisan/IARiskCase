from copy import copy
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import random
import time
import matplotlib.pyplot as plt
# assumptions
# 1. returns for each day is independent of return from previous day
# since there are days when US market is open and Canada isn't and vice versa to simplify program will only use
# dates where returns can be calculated for both markets. Future improvement would be use linear interpolation of
# asset price over missing date then calculate return to fill gaps in data.
# 2. returns for first day aren't present since close price is being so the first calculated return in day 2. Future
# improvement would be to updated script to seek out first close price preceding the start date used.

#variables
datefromat = '%Y-%m-%d'
startDateData = dt.date(2017,1,1).strftime(datefromat)      #start date of historical data
endDateData = dt.date(2021,12,31).strftime(datefromat)      #End date of historical data
forecastStartDate = dt.date(2022,3,31).strftime(datefromat)     #start date for forecase
iteration = 5        #Number of iterations (trials)
numprd = 4              #Number of periods stich together
szprd = 2              #size of periods, since 250 buisness days in a year approx 250/4=62.5
identiferCCy = []

# Function to load portfolio position and return dataframe to be used in risk analysis
def load_portfolio(file):
    # identifier -> how product is referenced
    # identifierType -> example ticker/ISIN/CUSIP
    # productType -> equity/bond/swap
    # baseCurrency -> currency in which product is priced in
    # positionValueCAD -> value of position converted to CAD (typically would get position value in base currency and
    # use current FX rate to convert. CAD simplifies example to maintain a starting allocation of 50% for both products)
    # build global array that has ticker and ccy pair to be able to apply fx correctly
    global identiferCCy
    df = pd.read_csv(file)
    identiferCCy = df[['identifier','baseCurrency']].to_numpy()
    return df



# return numpy array of daily close price for currency pair
def fx_rate(pair, startDate, endDate):
    return yf.Ticker(pair).history(start= startDate, end= endDate)["Close"].to_numpy()

# build dataframe for each type of risk factor with daily returns
# input start and end of date for historical data set and dataframe with only equities, returns a dataframe with the
# daily returns for each equity apart from first date
def equity_daily_return(port, startDate, endDate):
    equityReturns = []
    equity = port.loc[portfolio['productType'] == "equity"]
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

# funtion that will return initial price in dataframe given portfolio input
def initial_price (port, date):
    # isolate the equities in portfolio
    equities = port.loc[portfolio['productType'] == "equity"]
    initialPrice = []
    # yahoo finance to get price on date T the end date needs to be date T+1
    dateP1 = (dt.datetime.strptime(date, datefromat) + dt.timedelta(days=1)).strftime(datefromat)
    # loop through equites and uses base currency to determine whether to add '.to' for canadian markets
    for i in range(0, equities.shape[0]):
        if equities[i:i + 1]['baseCurrency'].values == 'CAD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0] + '.to').history(start=date,end=dateP1)["Close"]
        elif equities[i:i + 1]['baseCurrency'].values == 'USD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0]).history(start=date,end=dateP1)["Close"]
        temp = pd.DataFrame({equities[i:i + 1]['identifier'].values[0]: price.values})
        # Use inner join due to Canada and US having different holidays want to ensure a full populated data set
        if len(initialPrice) == 0:
            initialPrice = temp
        else:
            initialPrice = pd.concat([initialPrice, temp], axis=1, join='inner')

    return initialPrice

def initial_fx (date):

    intailFX = []
    # calculate CAD/USD rate using ((EUR/CAD)^-1)*(EUR/USD)
    cadusdinitial = np.multiply(np.reciprocal(fx_rate('EURUSD=X', date, date)),
                               fx_rate('EURCAD=X', date, date))
    temp = pd.DataFrame({'cadusd': cadusdinitial})
    # setup incase more currency pairs need to be added
    if len(intailFX) == 0:
        initialFX = temp
    else:
        initialFX = pd.concat([intailFX, temp], axis=1, join='inner')
    return initialFX

# funtion that will accept daily return and initial values of risk factors to return the forecasted results using the
# following method
# select m number of random sets of n returns and stitch returns together and return final values of factors
def forecast_risk_factor(initialValueEquity, dailyReturnEquity, initialValueFX, dailyReturnFX, m, n):
    # returns for FX and assets kept seperate
    forecast = np.zeros((m*n + 1, dailyReturnEquity.shape[1]))
    forecastFX = np.zeros((m*n + 1, dailyReturnFX.shape[1]))
    forecast[0] = initialValueEquity.values
    forecastFX[0] = initialValueFX.values
    for i in range(0,m):
        # randomly select value where set + start location won't be larger than array
        start = random.randint(0, dailyReturnEquity.shape[0] - n)
        offset = np.where(forecast == 0)[0][0]-1
        # apply return to value of previous day to calculate next day
        for j in range(start, start+n):
            forecast[j+1-start+offset] = forecast[j-start+offset]*dailyReturnEquity.values[j]
            forecastFX[j + 1 - start + offset] = forecastFX[j - start + offset] * dailyReturnFX.values[j]
    return forecast.transpose(), forecastFX.transpose()



portfolio = load_portfolio('portfolio.csv')        #load portfolio from csv
equityReturns = equity_daily_return(portfolio,startDateData,endDateData)        # generate equity returns
fxDailyChange = fx_daily_change(startDateData, endDateData)         # generate fx daily returns

# some FX data points missing from yahoo finance, for example 2017-11-16, using inner join to ensure data avaliable for
# all dates used in forecast
riskFactorInnerJoin = pd.concat([equityReturns, fxDailyChange], axis=1, join='inner')
equityReturns = riskFactorInnerJoin[equityReturns.columns]
fxDailyChange = riskFactorInnerJoin[fxDailyChange.columns]

#initial price of risk factors
initialPrice = initial_price(portfolio, forecastStartDate)
initialFX = initial_fx(forecastStartDate)

initialPos = initialPrice[portfolio['identifier'].values]   # calculated intial stocks help better visually stock price
for i in initialPos.columns:
    # attempt to calculate position only if header name exists in identifier ccy pair table, this ensures to not do
    # calculation on fx rate
    try:
        if identiferCCy[np.argwhere(identiferCCy == i)[0][0]][1] == 'USD':
            initialPos.at[0,i] *= initialFX.at[0,'cadusd']
        # get index of ticker in portfolio
        k = portfolio[portfolio['identifier'] == i].index.values[0]
        initialPos.at[0,i] = portfolio.at[k,'positionValueCAD']/initialPos.at[0,i]

    except:
        pass


# add checks to ensure riskfactors exist for each initial price

# forcast results
numAssets = equityReturns.shape[1]      # total number of assets in portfolio
forecasts = np.zeros((iteration*numAssets,numprd*szprd+1))      # forecast price change for assets
forecastsFX = np.zeros((iteration*fxDailyChange.shape[1],numprd*szprd+1))   # forecast price change for FX
for i in range(0, iteration):
    forecasts[i*numAssets:i*numAssets+numAssets], forecastsFX[i:i+1] = forecast_risk_factor(initialPrice, equityReturns, initialFX, fxDailyChange, numprd, szprd)

portfolioValue = np.zeros((iteration,numprd*szprd+1))       # determine value of portfolio over time
for i in range(0,iteration):
    forecastsTemp = forecasts[i*numAssets:i*numAssets+numAssets]
    # apply fx to USD stocks using the shape of portfolio to determine which columns refer to US stocks
    forecastsTemp[(portfolio['baseCurrency'].values == 'USD')] = forecastsTemp[(portfolio['baseCurrency'].values == 'USD')]*forecastsFX[i]
    tempinitialVal = (initialPos.values).flatten()
    for j in range(0,numprd*szprd+1):
        # add value of each asset * intial position to total portfolio value
        for k in range(0, numAssets):
            portfolioValue[i][j] += forecastsTemp[k][j]*tempinitialVal[k]

portfolioReturns = np.zeros((iteration,numprd*szprd))       # daily portfolio returns
portfolioReturnOverTime = np.zeros((iteration,numprd*szprd))    # compare return for each timeperiod compared to initial
for i in range(0,iteration):
    for j in range(0, numprd * szprd):
        portfolioReturns[i][j] = portfolioValue[i][j+1]/portfolioValue[i][j] - 1
        portfolioReturnOverTime[i][j] = portfolioValue[i][j+1]/portfolioValue[i][0] - 1


# Calculate the average return of portfolio
print('Average expected return: ', round(np.average(portfolioReturnOverTime[:,-1]),4)*100,'%')
# Calculate Expected 1% CVaR (Expected shortfall)
# np.sort(portfolioTotalReturn)
print('1% CVar: ', round(np.average(np.percentile(portfolioTotalReturn,1)),4)*100,'%')

# alternative tail risk, 1% tail risk based maxmium losses on indiviudal paths
print('%1 ETL based on largest single day loss per path ', round(np.average(np.percentile(portfolioReturns.min(axis=1),1)),4)*100,'%')

# alternative drawdown tail risk, typically based on largest peak to trough but here will be looking at largest decrease
# from initial capital. 1% tail risk based maximum losses on individual paths
print('%1 ETL based on largest over decrease from initial ', round(np.average(np.percentile(portfolioReturnOverTime.min(axis=1),1)),4)*100,'%')

print(portfolio)