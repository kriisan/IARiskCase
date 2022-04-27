import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import random

################################################################
# Assumptions
# 1. The return for each day is independent of the previous day. Provide returns are independent, to ensure data set is
# complete, there is a return value for each date used, I will be deleting  dates that dont have returns for each asset
# and fx. For future improvements I would look to use linear interpolation of prices over missing dates then calculate
# return to fill gaps in data to have more data points.
# 2. Returns for START_DATE_DATA aren't present since close price is being to calculate returns. Future improvement
# would be to have script look for the close price immediately preceding the START_DATE_DATA so it can be used to
# calculate the return
###############################################################

# Variables
DATE_FORMAT = '%Y-%m-%d'
START_DATE_DATA = dt.date(2017, 1, 1).strftime(DATE_FORMAT)  # Start date of historical data
END_DATE_DATA = dt.date(2021, 12, 31).strftime(DATE_FORMAT)  # End date of historical data
FORECAST_START_DATE = dt.date(2022, 3, 31).strftime(DATE_FORMAT)  # Start date for forecase
ITERATION = 5000  # Number of interations (trials)
NUM_PRD = 4  # Number of periods stich together
SZ_PRD = 62  # Size of periods, since 250 buisness days in a year approx 250/4=62.5
IDENTIFER_CCy = []


# Function to load portfolio position and return dataframe to be used in risk analysis
def load_portfolio(file):
    # Identifier -> how product is referenced
    # IdentifierType -> example ticker/ISIN/CUSIP
    # ProductType -> equity/bond/swap
    # BaseCurrency -> currency in which product is priced in
    # PositionValueCAD -> value of position converted to CAD (typically would get position value in base currency and
    # Use current FX rate to convert. CAD simplifies example to maintain a starting allocation of 50% for both products)
    # Build global array that has ticker and ccy pair to be able to apply fx correctly
    global IDENTIFER_CCy
    df = pd.read_csv(file)
    IDENTIFER_CCy = df[['identifier', 'baseCurrency']].to_numpy()
    return df


# Return numpy array of daily close price for currency pair
def fx_rate(pair, startDate, endDate):
    return yf.Ticker(pair).history(start=startDate, end=endDate)["Close"].to_numpy()


# Build dataframe for each type of risk factor with daily returns
# Input start and end of date for historical data set and dataframe with only equities, returns a dataframe with the
# daily returns for each equity apart from first date
def equity_daily_return(port, startDate, endDate):
    equityReturns = []
    equity = port.loc[portfolio['productType'] == "equity"]
    # Yahoo finance to get price on date T the end date needs to be date T+1
    endDate = (dt.datetime.strptime(endDate, DATE_FORMAT) + dt.timedelta(days=1)).strftime(DATE_FORMAT)
    for i in range(0, equity.shape[0]):
        # Get historical price data
        # Add error catching for tickers that don't work
        historicalPrice = []
        try:
            if equity[i:i + 1]['baseCurrency'].values == 'CAD':
                historicalPrice = \
                    yf.Ticker(equity[i:i + 1]['identifier'].values[0] + '.to').history(start=startDate, end=endDate)[
                        "Close"]
            elif equity[i:i + 1]['baseCurrency'].values == 'USD':
                historicalPrice = \
                    yf.Ticker(equity[i:i + 1]['identifier'].values[0]).history(start=startDate, end=endDate)["Close"]
        except:
            print('not valid ticker')
        # Calculate daily return current close price divided by previous day close price
        historicalPriceValue = historicalPrice.to_numpy()
        daily_return = [historicalPriceValue[i] / historicalPriceValue[i - 1] for i in
                        range(1, len(historicalPriceValue))]
        # Added temp filler 0 to first row of dataset to allow for date index to remain intact, added to beginning since
        # Return formula uses close price so daily return on start date isn't calculated
        daily_return.insert(0, 0)
        temp = pd.DataFrame({equity[i:i + 1]['identifier'].values[0]: daily_return}, index=historicalPrice.index).iloc[
               1:, :]
        # If first set of returns then set it as value of result else inner join with existing results
        if len(equityReturns) == 0:
            equityReturns = temp
        else:
            # Canada and US have different holidays so daily returns missing for certain, using inner join to ensure
            # dates used all have return values
            equityReturns = pd.concat([equityReturns, temp], axis=1, join='inner')

    return equityReturns


# Input start and end of date for historical data set, returns a dataframe with the daily returns for CAD/USD rate apart
# from return of day 1
def fx_daily_change(startDate, endDate):
    # Get currency pairs for dates used base/quote
    eurcad = yf.Ticker('EURCAD=X').history(start=startDate, end=endDate)["Close"]
    eurusd = yf.Ticker('EURUSD=X').history(start=startDate, end=endDate)["Close"]
    # Ensure fx rate available for all dates used
    tempFX = pd.concat([eurcad, eurusd], axis=1, join='inner')
    tempFX.columns = ['EURCAD', 'EURUSD']
    # Calculate CAD/USD rate using ((EUR/CAD)^-1)*(EUR/USD)
    cadusd = np.multiply(np.power(tempFX['EURUSD'].values, -1), tempFX['EURCAD'].values)
    # Calculate daily return current close price divided by previous day close price
    daily_return = [cadusd[i] / cadusd[i - 1] for i in range(1, len(cadusd))]
    # Added temp filler 0 to first row of dataset to allow for date index to remain intact, added to beginning since
    # Return formula uses close price so daily return on start date isn't calculated
    daily_return.insert(0, 0)
    # Convert result into dataframe
    fxDailyChange = pd.DataFrame({'cadusd': daily_return}, index=tempFX.index).iloc[1:, :]
    return fxDailyChange


# Function that will return initial price in dataframe given portfolio input
def initial_price(port, date):
    # Isolate the equities in portfolio
    equities = port.loc[portfolio['productType'] == "equity"]
    initialPrice = []
    # Yahoo finance to get price on date T the end date needs to be date T+1
    dateP1 = (dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1)).strftime(DATE_FORMAT)
    # Loop through equites and uses base currency to determine whether to add '.to' for canadian markets
    price = []
    for i in range(0, equities.shape[0]):
        if equities[i:i + 1]['baseCurrency'].values == 'CAD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0] + '.to').history(start=date, end=dateP1)[
                "Close"]
        elif equities[i:i + 1]['baseCurrency'].values == 'USD':
            price = yf.Ticker(equities[i:i + 1]['identifier'].values[0]).history(start=date, end=dateP1)["Close"]
        temp = pd.DataFrame({equities[i:i + 1]['identifier'].values[0]: price.values})
        # Use inner join due to Canada and US having different holidays want to ensure a full populated data set
        if len(initialPrice) == 0:
            initialPrice = temp
        else:
            initialPrice = pd.concat([initialPrice, temp], axis=1, join='inner')

    return initialPrice


# Return fx rate dataframe for given date
def initial_fx(date):
    intailFX = []
    # Calculate CAD/USD rate using ((EUR/CAD)^-1)*(EUR/USD)
    cadusdinitial = np.multiply(np.reciprocal(fx_rate('EURUSD=X', date, date)),
                                fx_rate('EURCAD=X', date, date))
    temp = pd.DataFrame({'cadusd': cadusdinitial})
    # Setup incase more currency pairs need to be added
    if len(intailFX) == 0:
        initialFX = temp
    else:
        initialFX = pd.concat([intailFX, temp], axis=1, join='inner')
    return initialFX


# Funtion that will accept daily return and initial values of risk factors to return the forecasted results using the
# following method
# Select m number of random sets of n returns and stitch returns together and return final values of factors
def forecast_risk_factor(initialValueEquity, dailyReturnEquity, initialValueFX, dailyReturnFX, m, n):
    # Returns for FX and assets kept separate
    forecast = np.zeros((m * n + 1, dailyReturnEquity.shape[1]))
    forecastFX = np.zeros((m * n + 1, dailyReturnFX.shape[1]))
    forecast[0] = initialValueEquity.values
    forecastFX[0] = initialValueFX.values
    for i in range(0, m):
        # Randomly select value where set + start location won't be larger than array
        start = random.randint(0, dailyReturnEquity.shape[0] - n)
        offset = np.where(forecast == 0)[0][0] - 1
        # Apply return to value of previous day to calculate next day
        for j in range(start, start + n):
            forecast[j + 1 - start + offset] = forecast[j - start + offset] * dailyReturnEquity.values[j]
            forecastFX[j + 1 - start + offset] = forecastFX[j - start + offset] * dailyReturnFX.values[j]
    return forecast.transpose(), forecastFX.transpose()


if __name__ == "__main__":
    print('Executing')
    portfolio = load_portfolio('portfolio.csv')  # load portfolio from csv
    equityReturns = equity_daily_return(portfolio, START_DATE_DATA, END_DATE_DATA)  # generate equity returns
    fxDailyChange = fx_daily_change(START_DATE_DATA, END_DATE_DATA)  # generate fx daily returns

    # Some FX data points missing from yahoo finance, for example 2017-11-16, using inner join to ensure data avaliable for
    # all dates used in forecast
    riskFactorInnerJoin = pd.concat([equityReturns, fxDailyChange], axis=1, join='inner')
    equityReturns = riskFactorInnerJoin[equityReturns.columns]
    fxDailyChange = riskFactorInnerJoin[fxDailyChange.columns]

    # Initial price of risk factors
    initialPrice = initial_price(portfolio, FORECAST_START_DATE)
    initialFX = initial_fx(FORECAST_START_DATE)

    initialPos = initialPrice[
        portfolio['identifier'].values]  # Calculated initial stocks help better visually stock price
    for i in initialPos.columns:
        # Attempt to calculate position only if header name exists in identifier ccy pair table, this ensures to not do
        # calculation on fx rate
        try:
            if IDENTIFER_CCy[np.argwhere(IDENTIFER_CCy == i)[0][0]][1] == 'USD':
                initialPos.at[0, i] *= initialFX.at[0, 'cadusd']
            # Get index of ticker in portfolio
            k = portfolio[portfolio['identifier'] == i].index.values[0]
            initialPos.at[0, i] = portfolio.at[k, 'positionValueCAD'] / initialPos.at[0, i]
        except:
            pass

    # Add checks to ensure riskfactors exist for each initial price

    # Forecast results
    numAssets = equityReturns.shape[1]  # Total number of assets in portfolio
    forecasts = np.zeros((ITERATION * numAssets, NUM_PRD * SZ_PRD + 1))  # Forecast price change for assets
    forecastsFX = np.zeros((ITERATION * fxDailyChange.shape[1], NUM_PRD * SZ_PRD + 1))  # Forecast price change for FX
    for i in range(0, ITERATION):
        forecasts[i * numAssets:i * numAssets + numAssets], forecastsFX[i:i + 1] = forecast_risk_factor(initialPrice,
                                                                                                        equityReturns,
                                                                                                        initialFX,
                                                                                                        fxDailyChange,
                                                                                                        NUM_PRD, SZ_PRD)

    portfolioValue = np.zeros((ITERATION, NUM_PRD * SZ_PRD + 1))  # Determine value of portfolio over time
    for i in range(0, ITERATION):
        forecastsTemp = forecasts[i * numAssets:i * numAssets + numAssets]
        # Apply fx to USD stocks using the shape of portfolio to determine which columns refer to US stocks
        forecastsTemp[(portfolio['baseCurrency'].values == 'USD')] = forecastsTemp[
                                                                         (portfolio['baseCurrency'].values == 'USD')] * \
                                                                     forecastsFX[i]
        tempinitialVal = (initialPos.values).flatten()
        for j in range(0, NUM_PRD * SZ_PRD + 1):
            # Add value of each asset * intial position to total portfolio value
            for k in range(0, numAssets):
                portfolioValue[i][j] += forecastsTemp[k][j] * tempinitialVal[k]

    portfolioReturns = np.zeros((ITERATION, NUM_PRD * SZ_PRD))  # Daily portfolio returns
    portfolioReturnOverTime = np.zeros(
        (ITERATION, NUM_PRD * SZ_PRD))  # Compare return for each timeperiod compared to initial
    for i in range(0, ITERATION):
        for j in range(0, NUM_PRD * SZ_PRD):
            portfolioReturns[i][j] = portfolioValue[i][j + 1] / portfolioValue[i][j] - 1
            portfolioReturnOverTime[i][j] = portfolioValue[i][j + 1] / portfolioValue[i][0] - 1

    # Calculate the average return of portfolio
    print('Average expected return: ', round(np.average(portfolioReturnOverTime[:, -1]), 4) * 100, '%')
    print('Std Dev of forecasted returns ', round(np.sqrt(np.var(portfolioReturnOverTime[:, -1])), 4))
    # Calculate Expected 1% CVaR (Expected shortfall)
    print('1% CVar: ', round(np.average(np.percentile(portfolioReturnOverTime[:, -1], 1)), 4) * 100, '%')

    # Alternative tail risk, 1% tail risk based maxmium losses on indiviudal paths
    print('%1 ETL based on largest single day loss per path ',
          round(np.average(np.percentile(portfolioReturns.min(axis=1), 1)), 4) * 100, '%')

    # Alternative risk metric drawdown, typically based on largest peak to trough but here will be looking at largest
    # decrease from initial capital. 1% tail risk based maximum losses on individual paths
    print('%1 ETL based on largest over decrease from initial ',
          round(np.average(np.percentile(portfolioReturnOverTime.min(axis=1), 1)), 4) * 100, '%')

    print(portfolio)
