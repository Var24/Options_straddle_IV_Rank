# Options Straddle strategy backtest code using IV Rank as signal

The IV rank, or Implied Volatility rank, is a metric used to check where the current implied volatility of a financial instrument stands relative to its recent trading history. 
It helps us assess whether the current implied volatility is high or low compared to past levels over a specified period. The period is considered as 252 days in our case.
We calculate the IV rank of SPX ATM options. The SPX options data is obtained from an online data vendor where daily options price, IV and greeks data are downloaded. The files
are stored as pickle files in the bz2 format.
The IV Rank calculation notebook calculates and saves the IV Rank data.
The short straddle notebook contains the code to back-test the options straddle wherein a straddle is initiated when the IV rank is greater than 50.
The exit signal is when the IV rank falls to 30 or the days left to expiry are 3. We don't trade on expiry days.
The back-test results are as follows:

![image](https://github.com/user-attachments/assets/bd577eca-eb82-4146-9a42-89ea2ac4b941)

The above results assume we are trading 5 lots of SPX.

The short straddle trading strategy has a win rate of 67.44% and the profit factor of 1.68. This indicates that the strategy has been profitable, with profits outweighing losses 
by some margin. But Per Trade PnL of Losers is 341 which is quite high. 
This means that we are losing $341 on every losing trade.
Further improvements can be made to this strategy by adding skew rank methodology. Also take profit and stop loss targets can be added as risk management tools to further improve 
the strategy.
