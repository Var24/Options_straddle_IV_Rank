# For data manipulation
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
import math

def trade_level_analytics(round_trips, lot_size):
    # Assume lot size as 5
    lot_size = 5

    # Calculate net premium
    round_trips['trade_wise_PnL'] = round_trips['Position'] * (round_trips['Exit_Price'] - round_trips['Entry_Price'])

    # Create a dataframe for storing trades
    trades = pd.DataFrame()

    # Groupby entry date
    trades_group = round_trips.groupby('Entry_Date')

    # Group trades from round_trips
    trades['Entry_Date'] = trades_group['Entry_Date'].first()
    trades['Exit_Date'] = trades_group['Exit_Date'].first()

    # Calculate PnL for the strategy for 1 lot
    trades['PnL'] = trades_group.trade_wise_PnL.sum() * lot_size

    # Calculate turnover for trades
    trades['Turnover'] = (trades_group['Exit_Price'].sum() + trades_group['Entry_Price'].sum()) * lot_size

    # Calculate PnL after deducting trading costs and slippages
    trades['PnL_post_trading_costs_slippages'] = trades['PnL'] - trades['Turnover'] * (0.01)

    # Create dataframe to store trade analytics
    analytics = pd.DataFrame(index=['Strategy'])

    # Calculate total PnL
    analytics['Total PnL'] = round(trades.PnL.sum(),2)

    # Number of total trades
    analytics['Total Trades'] = int(len(trades))

    # Profitable trades
    analytics['Number of Winners'] = int(len(trades.loc[trades.PnL > 0]))

    # Loss-making trades
    analytics['Number of Losers'] = int(len(trades.loc[trades.PnL <= 0]))

    # Win percentage
    analytics['Win (%)'] = round(100 * analytics['Number of Winners'] / analytics['Total Trades'],2)

    # Loss percentage
    analytics['Loss (%)'] = round(100 * analytics['Number of Losers'] / analytics['Total Trades'],2)

    # Per trade profit/loss of winning trades
    analytics['Per Trade PnL of Winners'] = round(trades.loc[trades.PnL > 0].PnL.mean(), 2)

    # Per trade profit/loss of losing trades
    analytics['Per Trade PnL of Losers'] = round(np.abs(trades.loc[trades.PnL <= 0].PnL.mean()), 2)

    # Calculate profit factor
    analytics['Profit Factor'] = round((analytics['Win (%)'] / 100 * analytics['Per Trade PnL of Winners']) / (
            analytics['Loss (%)'] / 100 * analytics['Per Trade PnL of Losers']), 2)

    return analytics.T


# Function for calculating mtm
def add_to_mtm(mark_to_market, option_strategy, trading_date):
    option_strategy['Date'] = trading_date
    mark_to_market = pd.concat([mark_to_market, option_strategy])
    return mark_to_market

# Function for fetching premium
def get_premium(options_strategy, options_data):

    # Get the premium for call option
    if options_strategy['Option Type'] == "CE":
        return options_data[' [C_LAST]']

    # Get the premium for put option
    elif options_strategy['Option Type'] == "PE":
        return options_data[' [P_LAST]']
    
# Function for fetching delta
def get_delta(options_strategy, options_data):
    
    # Get the delta for call option
    if options_strategy['Option Type'] == "CE":
        return options_data[' [C_DELTA]']

    # Get the delta for put option
    elif options_strategy['Option Type'] == "PE":
        return options_data[' [P_DELTA]']

# Function for setting up a straddle
def setup_straddle(options_data, direction='short'):

    # Create a dataframe to store the straddle
    straddle = pd.DataFrame()

    # CE and PE legs of the straddle
    straddle['Option Type'] = ['CE', 'PE']

    # Create the straddle at ATM
    straddle['Strike Price'] = options_data.ATM[0]

    # Sell positions for both CE and PE legs in case of a short straddle
    straddle['Position'] = -1
    
    # Buy positions for both CE and PE legs in case of a long straddle
    if direction == 'long':
        straddle['Position'] = 1
        
    # Get the premiums for the two option legs of the straddle
    straddle['Premium'] = straddle.apply(lambda r: get_premium(r, options_data), axis=1)

    # Get the delta values for the two option legs of the straddle
    straddle['Delta'] = straddle.apply(lambda r: get_delta(r, options_data), axis=1)

    return straddle

# Function to calculate PnL from the underlying position
def calculate_underlying_pnl(today_price, initial_price, position):
    return (today_price - initial_price) * position  # PnL = (current price - initial price) * position size

# Function to calculate the mark-to-market value of adjustments
def calculate_adjustment_mtm(row, underlying_price):
    return (underlying_price - row['Entry_Price']) * row['Position']


def sensitivity_analysis(options_data, delta):
    if delta == 'No Delta Hedging':
        delta = 100

    # Initialize dataframes for round trips, trades, and mark-to-market
    round_trips_details = pd.DataFrame()  # To record details of each round trip trade
    trades = pd.DataFrame()  # To record individual trades
    mark_to_market = pd.DataFrame()  # To record mark-to-market valuations

    # Initialize variables for current position, trade count, cumulative PnL, and underlying position
    current_position = 0  # Initial position is zero
    trade_num = 0  # Trade count starts at zero
    cum_pnl = 0  # Cumulative PnL starts at zero
    underlying_position = 0  # No initial position in the underlying asset
    initial_underlying_price = None  # Initial price of the underlying asset (to be set later)
    exit_flag = False  # Flag to indicate whether to exit a position

    # Set the start date for backtesting
    start_date = options_data.index[0]  # Use the first date in the options data index as the start date

    # Initialize adjustments DataFrame
    adjustments = pd.DataFrame(columns=['Trade_Num', 'Entry_Date', 'Position', 'Delta', 'Entry_Price'])

    for i in options_data.loc[start_date:].index.unique():
        today_data = options_data.loc[i]  # Get the current day options data
        underlying_price = today_data[' [UNDERLYING_LAST]'].iloc[0]  # Get the underlying price for the current day

        if current_position == 1:  # If we have an open position
            setup_strike_data = today_data[today_data[' [STRIKE]'] == setup_strike]  # Get data for the current strike price
            straddle['Premium'] = straddle.apply(lambda r: get_premium(r, setup_strike_data), axis=1)  # Calculate premium for straddle
            straddle['Delta'] = straddle.apply(lambda r: get_delta(r, setup_strike_data), axis=1)  # Calculate delta for straddle
            
            net_premium = (straddle.Position * straddle.Premium).sum()  # Calculate net premium of the straddle

            # Calculate PnL from the underlying position
            if initial_underlying_price is not None:
                underlying_pnl = calculate_underlying_pnl(underlying_price, initial_underlying_price, underlying_position)
            else:
                underlying_pnl = 0  # Set PnL to 0 if there's no initial price

            # Update the mark-to-market value with the straddle
            mark_to_market = add_to_mtm(mark_to_market, straddle, i)

            # Include adjustments in the mark-to-market
            if not adjustments.empty:
                adjustments['Mark_to_Market'] = adjustments.apply(
                    lambda row: calculate_adjustment_mtm(row, underlying_price), axis=1
                )
                # Add adjustments to mark-to-market
                for index, row in adjustments.iterrows():
                    adjustment_entry = {
                        'Asset Type': row['Asset Type'],
                        'Strike Price': 'NA',
                        'Position': row['Position'],
                        'Premium': row['Mark_to_Market'],
                        'Delta': row['Delta'],
                        'Date': i
                    }
                    mark_to_market = mark_to_market.append(adjustment_entry, ignore_index=True)
                    
                adjustments.drop('Mark_to_Market', axis=1, inplace=True)


            # Check if any exit conditions are met
            if today_data['Signal'].iloc[0] == 0:  # Exit signal is 0
                exit_type = today_data['Exit_Type'].iloc[0]                
                exit_flag = True

            if exit_flag:
                trades['Exit_Date'] = i  # Record exit date
                trades['Exit_Type'] = exit_type  # Record exit type
                trades['Exit_Price'] = straddle.Premium  # Record exit price

                # Ensure there are adjustments for the current trade
                if not adjustments.empty and trade_num in adjustments['Trade_Num'].values:
                    # Update exit details for all adjustments related to the current trade
                    adjustments.loc[adjustments['Trade_Num'] == trade_num, 'Exit_Date'] = i
                    adjustments.loc[adjustments['Trade_Num'] == trade_num, 'Exit_Price'] = underlying_price

                net_premium = round((straddle.Position * straddle.Premium).sum(), 1)  # Calculate net premium again
                entry_net_premium = (trades.Position * trades.Entry_Price).sum()  # Calculate entry net premium
                trades['PnL'] = trades['Entry_Price'] - trades['Exit_Price']  # Calculate PnL for the trades

                if initial_underlying_price is not None and adjustments.empty:
                    delta_trade = pd.DataFrame({
                        'Asset Type': "Underlying",
                        'Strike Price': 'NA',
                        'Position': round(underlying_position, 2),
                        'Delta': round(underlying_position, 2),
                        'Entry_Date': initial_underlying_date,  # Record initial date
                        'Exit_Date': [i],
                        'Exit_Type': 'Delta Hedge',  # Record exit type as delta hedge
                        'Entry_Price': initial_underlying_price,  # Record initial price
                        'Exit_Price': underlying_price,
                        'PnL': (underlying_price - initial_underlying_price) * underlying_position  # Record underlying PnL
                    })
                    trades = pd.concat([trades, delta_trade])  # Concatenate the new trade details

                trade_pnl = round(net_premium - entry_net_premium, 1)   # Calculate total PnL for the trade
                round_trips_details = pd.concat([round_trips_details, trades])  # Record the round trip details

                # Append adjustments to round trips details
                if not adjustments.empty and trade_num in adjustments['Trade_Num'].values:
                    adjustments['PnL'] = (adjustments['Exit_Price'] - adjustments['Entry_Price']) * adjustments['Position']
                    round_trips_details = pd.concat([round_trips_details, adjustments[adjustments['Trade_Num'] == trade_num]])
                    trade_pnl = trade_pnl + adjustments.PnL.sum()
                    
                cum_pnl += trade_pnl  # Update cumulative PnL
                cum_pnl = round(cum_pnl, 2)  # Round cumulative PnL
                current_position = 0  # Reset current position
                initial_underlying_price = None  # Reset initial underlying price
                initial_underlying_date = None  # Reset initial underlying date
                underlying_position = 0  # Reset the underlying position
                exit_flag = False  # Reset exit flag

                # Clear adjustments for the current trade
                adjustments = adjustments[adjustments['Trade_Num'] != trade_num]

            else:  # If no exit conditions are met
                # Calculate net delta including the underlying position
                try:
                    straddle_delta = (straddle.Position * straddle.Delta).sum()
                    underlying_delta = underlying_position
                    net_delta = straddle_delta + underlying_position
                except KeyError:
                    #print(f"Data missing for the required strike prices on {i}, Not adding to trade logs.")
                    current_position = 0  # Reset position if data is missing
                    continue

                if abs(net_delta) > delta:  # Check if net delta exceeds the threshold
                    adjustment = -net_delta  # Calculate adjustment needed
                    underlying_position += adjustment  # Adjust the underlying position
                    initial_underlying_price = underlying_price

                    delta_trade = pd.DataFrame({
                        'Trade_Num': trade_num,
                        'Asset Type': 'Underlying',
                        'Strike Price': 'NA',
                        'Entry_Date': [i],  # Date of adjustment
                        'Position': round(adjustment, 2),
                        'Delta': round(adjustment, 2),
                        'Entry_Price': initial_underlying_price,  # Price at adjustment
                        'Exit_Type': "Delta Hedge"
                    })
                    
                    adjustments = pd.concat([adjustments, delta_trade], ignore_index=True)  # Add details to adjustments
                    #print(f"Delta Hedge | Date: {i} | Straddle Delta: {round(straddle_delta, 2)} | | Underlying Delta: {round(underlying_delta, 2)} | | Net Delta: {round(net_delta, 2)} | Adjustment: {round(adjustment, 2)}")
                    

        if current_position == 0 and today_data['Signal'].iloc[0] == 1:  # Entry signal is 1
            trade_pnl = 0  # Reset trade PnL
            setup_strike_data = today_data[today_data['ATM'] == today_data[' [STRIKE]']]  # Get data for ATM strike

            straddle = setup_straddle(setup_strike_data, direction="short")  # Setup a new short straddle
            setup_strike = today_data['ATM'].iloc[0]  # Set the strike price

            trades = straddle.copy()  # Copy straddle details to trades
            trades['Entry_Date'] = i  # Record entry date
            trades.rename(columns={'Premium': 'Entry_Price'}, inplace=True)  # Rename premium column to entry price

            net_premium = round((straddle.Position * straddle.Premium).sum(), 2)  # Calculate net premium

            try:
                net_delta = round((straddle.Position * straddle.Delta).sum(),2)  # Calculate net delta
            except KeyError:
                #print(f"Data missing for the required strike prices on {i}, Not adding to trade logs.")
                current_position = 0  # Reset position if data is missing
                continue

            current_position = 1  # Set current position to 1 (open)
            mark_to_market = add_to_mtm(mark_to_market, straddle, i)  # Add to mark-to-market

            trade_num += 1  # Increment trade number
            
            trades['Trade_Num'] = trade_num
            #print("-" * 30)
            #print(f"Trade No: {trade_num} | Entry | Date: {i} | Premium: {net_premium} | Initial Delta: {round(net_delta, 2)} | Trade PnL: {trade_pnl} | Cum PnL: {cum_pnl}")


    return [round_trips_details.PnL.sum(), len(round_trips_details)]
