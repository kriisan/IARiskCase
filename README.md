# OTPP_IARiskCase

This program was developed to forecast returns of a equity portfolio using historical data. User can define the size and number of steps that will be stitched together to generate the forecasted returns for the given portfolio. The results will provide expected return, CVaR and a couple other risk metrics.

To execute the program you please look at the below steps:
1.	Update the ‘portfolio.csv’ file with CAD and USD equities from your portfolio. Ensure that all date points are populated for each asset (identifier, identifierType, productType, baseCurrency, positionValueCAD). Ensure the file is in the folder as RiskModel.py
2.	Update the variables in found near the top of the RiskModel.py script to change risk model parameters
3.	Execute the RiskModel.py script

Inputs:
portfolio.csv

Outputs:
average return,
standard deviation of returns,
1% CVaR of returns,
1%CVaR of largest inpath difference from initial amount
